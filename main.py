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

    MODIFICADO: Usa Flood Fill para una selección de fondo contigua.
    """
    # Convertir a HSV para la detección de color
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    # Punto de semilla para floodFill (ej. esquina superior izquierda, o un punto configurable)
    # Asumimos que el fondo está en (0,0) o (5,5) para evitar bordes puros.
    # Podrías hacerlo configurable si el fondo no siempre está en la esquina.
    seed_point = (5, 5) 

    # Crear una máscara temporal para el floodFill (debe ser 2 píxeles más grande que la imagen)
    # y de tipo CV_8UC1 (np.uint8).
    h, w, _ = image_np.shape
    mask_floodfill = np.zeros((h + 2, w + 2), np.uint8)

    # El color del píxel inicial para floodFill
    seed_color_hsv = hsv_image[seed_point[1], seed_point[0]] # [y, x]

    # Calcular loDiff y upDiff para floodFill
    # loDiff y upDiff son las diferencias máximas *hacia abajo* y *hacia arriba*
    # en los componentes del color (H, S, V) desde el color del punto de semilla.
    # Para usar tus rangos lower_hsv/upper_hsv con floodFill, calculamos la diferencia
    # entre el color del punto de semilla y tus límites.

    # Aquí hay una forma de adaptar tus rangos a loDiff/upDiff:
    # Asegúrate de que los valores no sean negativos
    loDiff_h = max(0, int(seed_color_hsv[0] - lower_hsv_h))
    loDiff_s = max(0, int(seed_color_hsv[1] - lower_hsv_s))
    loDiff_v = max(0, int(seed_color_hsv[2] - lower_hsv_v))

    upDiff_h = max(0, int(upper_hsv_h - seed_color_hsv[0]))
    upDiff_s = max(0, int(upper_hsv_s - seed_color_hsv[1]))
    upDiff_v = max(0, int(upper_hsv_v - seed_color_hsv[2]))

    loDiff = (loDiff_h, loDiff_s, loDiff_v)
    upDiff = (upDiff_h, upDiff_s, upDiff_v)

    # Realizar el relleno por inundación
    # Valor de relleno (aquí, 255 para marcar el área de fondo en la máscara)
    # FLODFILL_MASK_ONLY: Solo modifica la máscara
    # FLODFILL_FIXED_RANGE: Usa loDiff/upDiff en relación al color del punto de semilla
    cv2.floodFill(hsv_image, mask_floodfill, seed_point, (0, 0, 0), loDiff=loDiff, upDiff=upDiff, flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)

    # La máscara resultante de floodFill está en mask_floodfill[1:-1, 1:-1]
    # Invertirla para que 255 sea el objeto (púa) y 0 sea el fondo
    mask_object = mask_floodfill[1:-1, 1:-1] # Extrae la parte de la máscara que corresponde a la imagen original
    mask_inverted = cv2.bitwise_not(mask_object) # Ahora 255 = objeto, 0 = fondo

    # 2. Desenfoque y canal alfa
    if feather_blur_kernel_size % 2 == 0 or feather_blur_kernel_size <= 0:
        feather_blur_kernel_size = 15 if feather_blur_kernel_size <= 0 else feather_blur_kernel_size + 1
    feather_blur_kernel_tuple = (feather_blur_kernel_size, feather_blur_kernel_size)
    
    # Aplicar el desenfoque a la máscara invertida (objeto) para suavizar bordes
    blurred_mask = cv2.GaussianBlur(mask_inverted, feather_blur_kernel_tuple, 0)
    
    # Interpolar el canal alfa
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
    return {"status": "✅ Servidor de recorte de fondo activo (método chroma key avanzado con flood fill)"}

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

# --- Endpoint de Procesamiento por Lotes ---
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
    elimina el fondo, detecta cada púa, y devuelve un archivo ZIP con cada púa
    recortada como un archivo WebP individual, manteniendo su orientación original.
    """
    try:
        # Leer la imagen A4
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_a4 = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Lee como BGR

        if img_a4 is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen A4.")

        # 1. Aplicar Chroma Key a toda la imagen A4
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
        
        output_images = []
        MIN_PUA_AREA = 12000  # Área mínima para considerar un contorno como una púa
        MAX_PUA_AREA = 30000  # Área máxima para evitar detectar la hoja completa

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if MIN_PUA_AREA < area < MAX_PUA_AREA: 
                # Obtener el rectángulo delimitador (axial) de la púa en la imagen original
                x, y, w, h = cv2.boundingRect(contour)

                # Añadir un pequeño padding al recorte final
                padding = 10 
                x_final = max(0, x - padding)
                y_final = max(0, y - padding)
                w_final = min(processed_a4_img_rgba.shape[1] - x_final, w + 2 * padding)
                h_final = min(processed_a4_img_rgba.shape[0] - y_final, h + 2 * padding)
                
                # Recortar la púa directamente de la imagen RGBA procesada
                pua_cropped_rgba = processed_a4_img_rgba[y_final : y_final + h_final, x_final : x_final + w_final]

                if pua_cropped_rgba.size == 0:
                    print(f"Advertencia: Recorte vacío para contorno {i+1}. Saltando.")
                    continue 

                # Convertir a WebP y añadir a la lista de salida
                is_success, buffer = cv2.imencode(".webp", pua_cropped_rgba)
                if is_success:
                    output_images.append({
                        "filename": f"{file_prefix}_{i+1}.webp",
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

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR en batch-process: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error interno del servidor al procesar por lotes: {e}")
