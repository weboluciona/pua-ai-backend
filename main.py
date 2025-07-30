import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Response, HTTPException, status, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import zipfile # Necesario para crear archivos ZIP

# 🚀 Inicializar FastAPI
app = FastAPI()

# 🛡️ Añadir CORS para que tu HTML pueda hacer peticiones sin bloqueo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://weboluciona.com"], # Puedes restringirlo a tu dominio si prefieres (ej. ["https://tudominio.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🎨 PARAMETROS CLAVE PARA AJUSTAR LA ELIMINACIÓN DEL FONDO (Valores por defecto si no se proporcionan)
DEFAULT_LOWER_HSV_H = 140
DEFAULT_LOWER_HSV_S = 50
DEFAULT_LOWER_HSV_V = 50
DEFAULT_UPPER_HSV_H = 170
DEFAULT_UPPER_HSV_S = 255
DEFAULT_UPPER_HSV_V = 255

DEFAULT_FEATHER_BLUR_KERNEL_SIZE = 19
DEFAULT_ALPHA_THRESHOLD_FG = 200
DEFAULT_ALPHA_THRESHOLD_BG = 160
DEFAULT_MORPH_KERNEL_SIZE = 5
DEFAULT_MORPH_ITERATIONS = 3

# --- Función de lógica principal de Chroma Key (REUTILIZABLE) ---
# Ahora devuelve la imagen RGBA PROCESADA y la MÁSCARA BINARIA para detección de contornos
def _apply_chroma_key_logic(
    image_np: np.ndarray, # Espera una imagen OpenCV (BGR)
    lower_hsv_h: int, lower_hsv_s: int, lower_hsv_v: int,
    upper_hsv_h: int, upper_hsv_s: int, upper_hsv_v: int,
    feather_blur_kernel_size: int,
    alpha_threshold_fg: int, alpha_threshold_bg: int,
    morph_kernel_size: int, morph_iterations: int
) -> tuple[np.ndarray, np.ndarray]: # Devuelve (imagen RGBA, máscara binaria para contornos)
    """
    Aplica el algoritmo de Chroma Key avanzado a una imagen y devuelve la imagen RGBA
    y una máscara binaria limpia para la detección de contornos.
    """
    # 1. Definir los límites HSV
    lower_hsv = np.array([lower_hsv_h, lower_hsv_s, lower_hsv_v])
    upper_hsv = np.array([upper_hsv_h, upper_hsv_s, upper_hsv_v])

    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    mask_inverted = cv2.bitwise_not(mask) # Máscara del objeto (púa)

    # 2. Desenfoque y canal alfa
    if feather_blur_kernel_size % 2 == 0 or feather_blur_kernel_size <= 0:
        feather_blur_kernel_size = 15 if feather_blur_kernel_size <= 0 else feather_blur_kernel_size + 1
    feather_blur_kernel_tuple = (feather_blur_kernel_size, feather_blur_kernel_size)
    blurred_mask = cv2.GaussianBlur(mask_inverted, feather_blur_kernel_tuple, 0)
    
    alpha_channel = np.interp(blurred_mask,
                              [alpha_threshold_bg, alpha_threshold_fg],
                              [0, 255]).astype(np.uint8)

    # 3. Combinar con la imagen original para obtener la imagen RGBA final
    b, g, r = cv2.split(image_np)
    rgba_image = cv2.merge([b, g, r, alpha_channel])

    # 4. Crear una máscara binaria limpia para encontrar contornos (usando el alpha_channel)
    # Aplicamos umbral y operaciones morfológicas para asegurar contornos bien definidos
    binary_mask_for_contours = alpha_channel.copy()
    _, binary_mask_for_contours = cv2.threshold(binary_mask_for_contours, 1, 255, cv2.THRESH_BINARY)
    
    # Operaciones morfológicas para limpiar y separar contornos antes de findContours
    if morph_kernel_size % 2 == 0 or morph_kernel_size <= 0:
        morph_kernel_size = 5 if morph_kernel_size <=0 else morph_kernel_size + 1
    kernel_morph_batch = np.ones((morph_kernel_size, morph_kernel_size), np.uint8) 
    
    binary_mask_for_contours = cv2.morphologyEx(binary_mask_for_contours, cv2.MORPH_CLOSE, kernel_morph_batch, iterations=morph_iterations)
    binary_mask_for_contours = cv2.morphologyEx(binary_mask_for_contours, cv2.MORPH_OPEN, kernel_morph_batch, iterations=morph_iterations)

    return rgba_image, binary_mask_for_contours

# 📸 Endpoint de salud para Render
@app.get("/healthz", summary="Health Check")
def health_check():
    """
    Endpoint para que Render y otros sistemas verifiquen el estado del servicio.
    """
    return {"status": "ok"}

# 🏠 Endpoint base por si visitas la raíz en el navegador
@app.get("/", summary="Root Endpoint")
def home():
    """
    Endpoint de bienvenida para el servicio de recorte de fondo.
    """
    return {"status": "✅ Servidor de recorte de fondo activo (método chroma key avanzado)"}

# 📸 Endpoint para procesar una sola imagen y quitar el fondo
@app.post("/procesar-foto", summary="Process single image and remove background with provided parameters")
async def procesar_foto(
    file: UploadFile = File(...),
    lower_hsv_h: int | None = Form(DEFAULT_LOWER_HSV_H),
    lower_hsv_s: int | None = Form(DEFAULT_LOWER_HSV_S),
    lower_hsv_v: int | None = Form(DEFAULT_LOWER_HSV_V),
    upper_hsv_h: int | None = Form(DEFAULT_UPPER_HSV_H),
    upper_hsv_s: int | None = Form(DEFAULT_UPPER_HSV_S),
    upper_hsv_v: int | None = Form(DEFAULT_UPPER_HSV_V),
    feather_blur_kernel_size: int | None = Form(DEFAULT_FEATHER_BLUR_KERNEL_SIZE),
    alpha_threshold_fg: int | None = Form(DEFAULT_ALPHA_THRESHOLD_FG),
    alpha_threshold_bg: int | None = Form(DEFAULT_ALPHA_THRESHOLD_BG),
    morph_kernel_size: int | None = Form(DEFAULT_MORPH_KERNEL_SIZE),
    morph_iterations: int | None = Form(DEFAULT_MORPH_ITERATIONS)
):
    """
    Recibe una imagen y TODOS los parámetros de configuración para chroma key desde el frontend/PHP,
    aplica el algoritmo, redimensiona a un máximo de 350px de ancho y devuelve la imagen con el fondo eliminado (WebP con transparencia).
    """
    try:
        input_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Aplicar la lógica de Chroma Key (solo necesitamos la imagen RGBA aquí)
        rgba_image, _ = _apply_chroma_key_logic( # Ignoramos la máscara binaria
            opencv_image,
            lower_hsv_h, lower_hsv_s, lower_hsv_v,
            upper_hsv_h, upper_hsv_s, upper_hsv_v,
            feather_blur_kernel_size,
            alpha_threshold_fg, alpha_threshold_bg,
            morph_kernel_size, morph_iterations
        )

        # Convertir a PIL Image para redimensionar y guardar como WebP
        output_pil_image = Image.fromarray(cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGBA))
        
        # --- Redimensionar la imagen si su ancho es mayor a 350px ---
        MAX_WIDTH_OUTPUT = 350
        if output_pil_image.width > MAX_WIDTH_OUTPUT:
            aspect_ratio = output_pil_image.height / output_pil_image.width
            new_height = int(MAX_WIDTH_OUTPUT * aspect_ratio)
            output_pil_image = output_pil_image.resize((MAX_WIDTH_OUTPUT, new_height), Image.LANCZOS)

        # Guardar en buffer como WebP
        output_buffer = io.BytesIO()
        output_pil_image.save(output_buffer, format="WEBP", quality=80, method=6) # method=6 para mejor compresión WebP
        output_buffer.seek(0)
        
        return Response(content=output_buffer.getvalue(), media_type="image/webp")

    except HTTPException: # Re-lanza HTTPExceptions ya definidas (ej. 400 Bad Request)
        raise
    except Exception as e:
        print(f"ERROR: Fallo al procesar la foto (chroma key OpenCV): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing image: {e}")

# --- NUEVA RUTA: Procesamiento por Lotes ---
@app.post("/batch-process", summary="Process A4 image with multiple picks and return a ZIP of cropped WebP images")
async def batch_process(
    file: UploadFile = File(...),
    lower_hsv_h: int | None = Form(DEFAULT_LOWER_HSV_H),
    lower_hsv_s: int | None = Form(DEFAULT_LOWER_HSV_S),
    lower_hsv_v: int | None = Form(DEFAULT_LOWER_HSV_V),
    upper_hsv_h: int | None = Form(DEFAULT_UPPER_HSV_H),
    upper_hsv_s: int | None = Form(DEFAULT_UPPER_HSV_S),
    upper_hsv_v: int | None = Form(DEFAULT_UPPER_HSV_V),
    feather_blur_kernel_size: int | None = Form(DEFAULT_FEATHER_BLUR_KERNEL_SIZE),
    alpha_threshold_fg: int | None = Form(DEFAULT_ALPHA_THRESHOLD_FG),
    alpha_threshold_bg: int | None = Form(DEFAULT_ALPHA_THRESHOLD_BG),
    morph_kernel_size: int | None = Form(DEFAULT_MORPH_KERNEL_SIZE),
    morph_iterations: int | None = Form(DEFAULT_MORPH_ITERATIONS),
    file_prefix: str = Form("pua") # Recibe el prefijo del frontend
):
    """
    Recibe una imagen A4 con múltiples púas en un fondo chroma,
    elimina el fondo, detecta cada púa, la rota para que quede vertical
    y devuelve un archivo ZIP con cada púa recortada como un archivo WebP individual.
    """
    try:
        # Leer la imagen A4
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_a4 = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Lee como BGR

        if img_a4 is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen A4.")

        # 1. Aplicar Chroma Key a toda la imagen A4
        # Ahora _apply_chroma_key_logic devuelve la imagen RGBA y la máscara binaria
        processed_a4_img_rgba, binary_mask_for_contours = _apply_chroma_key_logic(
            img_a4,
            lower_hsv_h, lower_hsv_s, lower_hsv_v,
            upper_hsv_h, upper_hsv_s, upper_hsv_v,
            feather_blur_kernel_size,
            alpha_threshold_fg, alpha_threshold_bg,
            morph_kernel_size, morph_iterations
        )

        # 2. Detección de Contornos (para identificar cada púa)
        contours, _ = cv2.findContours(binary_mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- AÑADIR ESTOS PRINTS TEMPORALES PARA DEPURACIÓN ---
        print("\n--- INICIO DE DEBUG: Áreas de Contornos Detectados ---")
        if not contours:
            print("No se detectaron contornos en la imagen A4. Intenta ajustar los parámetros HSV o los parámetros morfológicos.")
        # --- FIN DE PRINTS TEMPORALES ---

        output_images = []
        MIN_PUA_AREA = 12000  # Área mínima para considerar un contorno como una púa
        MAX_PUA_AREA = 30000  # Área máxima para evitar detectar la hoja completa

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            print(f"Contorno {i+1}: Área = {int(area)} píxeles cuadrados") # Debugging
            
            if MIN_PUA_AREA < area < MAX_PUA_AREA: 
                # Obtener el rectángulo de área mínima y su ángulo
                rect = cv2.minAreaRect(contour)
                (center_x, center_y), (width, height), angle = rect

                # --- Lógica de Rotación para Verticalidad ---
                # Ajustar el ángulo para que el lado más largo sea la "altura"
                if width < height: # Si el 'height' es el lado más largo
                    # El ángulo ya está referenciado al eje Y (vertical)
                    # minAreaRect da ángulos en [-90, 0) para rectángulos más altos que anchos.
                    # Queremos que la púa quede con 0 grados (vertical)
                    rotation_angle = angle
                else: # Si el 'width' es el lado más largo (rectángulo "acostado")
                    # El ángulo está referenciado al eje X (horizontal)
                    # Sumamos 90 para referenciarlo al eje Y (vertical)
                    rotation_angle = angle + 90
                
                # Para asegurar que la púa no quede "boca abajo"
                # Esto es una heurística y puede requerir ajuste fino si las púas tienen una orientación preferida
                # (ej. la punta siempre hacia abajo).
                # Por ahora, simplemente rota para que el lado más largo sea vertical.
                
                # Crear la matriz de rotación
                M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
                
                # Calcular las nuevas dimensiones de la imagen después de rotar para evitar recortes
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_width = int((processed_a4_img_rgba.shape[1] * cos) + (processed_a4_img_rgba.shape[0] * sin))
                new_height = int((processed_a4_img_rgba.shape[1] * sin) + (processed_a4_img_rgba.shape[0] * cos))

                # Ajustar la matriz de rotación para trasladar la imagen al nuevo centro
                M[0, 2] += (new_width / 2) - center_x
                M[1, 2] += (new_height / 2) - center_y

                # Rotar la imagen RGBA completa
                rotated_image_full = cv2.warpAffine(processed_a4_img_rgba, M, (new_width, new_height), 
                                                    flags=cv2.INTER_LANCZOS4, 
                                                    borderMode=cv2.BORDER_CONSTANT, 
                                                    borderValue=(0, 0, 0, 0)) # Fondo transparente

                # Transformar el contorno original con la misma matriz de rotación
                # Esto nos da la nueva posición del contorno en la imagen rotada
                transformed_contour = cv2.transform(contour.reshape(-1, 1, 2), M).reshape(-1, 1, 2)
                
                # Obtener el nuevo rectángulo delimitador (axial) de la púa en la imagen rotada
                x_rot, y_rot, w_rot, h_rot = cv2.boundingRect(transformed_contour)

                # Añadir un pequeño padding al recorte final
                padding = 10 
                x_final = max(0, x_rot - padding)
                y_final = max(0, y_rot - padding)
                w_final = min(rotated_image_full.shape[1] - x_final, w_rot + 2 * padding)
                h_final = min(rotated_image_full.shape[0] - y_final, h_rot + 2 * padding)
                
                # Recortar la púa de la imagen rotada completa
                pua_cropped_rgba = rotated_image_full[y_final : y_final + h_final, x_final : x_final + w_final]

                if pua_cropped_rgba.size == 0:
                    print(f"Advertencia: Recorte vacío para contorno {i+1}. Saltando.")
                    continue 

                # Convertir a WebP y añadir a la lista de salida
                is_success, buffer = cv2.imencode(".webp", pua_cropped_rgba)
                if is_success:
                    output_images.append({
                        "filename": f"{file_prefix}_{i+1}.webp", # Usamos el prefijo aquí
                        "data": io.BytesIO(buffer.tobytes())
                    })
        
        if not output_images:
            raise HTTPException(status_code=404, detail=f"No se detectaron púas válidas para procesar con los parámetros dados. Verifique el rango de área ({MIN_PUA_AREA}-{MAX_PUA_AREA}) o los parámetros de Chroma Key.")

        # 3. Comprimir todas las imágenes en un archivo ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for img_info in output_images:
                zf.writestr(img_info["filename"], img_info["data"].getvalue())
        zip_buffer.seek(0)
        
        # Define el nombre del archivo ZIP
        zip_filename = f"{file_prefix}_puas_procesadas.zip"

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )

    except HTTPException: # Re-lanza HTTPExceptions ya definidas
        raise
    except Exception as e:
        print(f"ERROR en batch-process: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error interno del servidor al procesar por lotes: {e}")
