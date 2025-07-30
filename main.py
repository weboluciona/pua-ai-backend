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
    allow_origins=["*"], # Puedes restringirlo a tu dominio si prefieres (ej. ["https://tudominio.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🎨 PARAMETROS CLAVE PARA AJUSTAR LA ELIMINACIÓN DEL FONDO (Valores por defecto si no se proporcionan)
# Estos valores por defecto son para el caso de que el frontend no los envíe,
# pero el frontend siempre los envía, por lo que son más bien una referencia.
DEFAULT_LOWER_HSV_H = 140
DEFAULT_LOWER_HSV_S = 50
DEFAULT_LOWER_HSV_V = 50
DEFAULT_UPPER_HSV_H = 170
DEFAULT_UPPER_HSV_S = 255
DEFAULT_UPPER_HSV_V = 255

DEFAULT_FEATHER_BLUR_KERNEL_SIZE = 19 # Ajustado a 19 como en el frontend
DEFAULT_ALPHA_THRESHOLD_FG = 200 # Ajustado a 200 como en el frontend
DEFAULT_ALPHA_THRESHOLD_BG = 160 # Ajustado a 160 como en el frontend
DEFAULT_MORPH_KERNEL_SIZE = 5
DEFAULT_MORPH_ITERATIONS = 3 # Ajustado a 3 como en el frontend

# --- Función de lógica principal de Chroma Key (REUTILIZABLE) ---
def _apply_chroma_key_logic(
    image_np: np.ndarray, # Espera una imagen OpenCV (BGR)
    lower_hsv_h: int, lower_hsv_s: int, lower_hsv_v: int,
    upper_hsv_h: int, upper_hsv_s: int, upper_hsv_v: int,
    feather_blur_kernel_size: int,
    alpha_threshold_fg: int, alpha_threshold_bg: int,
    morph_kernel_size: int, morph_iterations: int
) -> np.ndarray: # Devuelve una imagen OpenCV (BGRA) con canal alfa
    """
    Aplica el algoritmo de Chroma Key avanzado a una imagen.
    """
    # 1. Definir los límites HSV
    lower_hsv = np.array([lower_hsv_h, lower_hsv_s, lower_hsv_v])
    upper_hsv = np.array([upper_hsv_h, upper_hsv_s, upper_hsv_v])

    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    mask_inverted = cv2.bitwise_not(mask)

    # 2. Operaciones morfológicas
    # Asegúrate de que los valores de kernel sean impares y positivos
    if morph_kernel_size % 2 == 0 or morph_kernel_size <= 0:
        morph_kernel_size = 5 if morph_kernel_size <=0 else morph_kernel_size + 1 # Asegura que sea impar y positivo
        print(f"Advertencia: morph_kernel_size ajustado a {morph_kernel_size} (debe ser impar y positivo).")

    kernel_morph = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    
    mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_OPEN, kernel_morph, iterations=morph_iterations)
    mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_CLOSE, kernel_morph, iterations=morph_iterations)

    # 3. Desenfoque y canal alfa
    # Asegúrate de que feather_blur_kernel_size sea impar y positivo
    if feather_blur_kernel_size % 2 == 0 or feather_blur_kernel_size <= 0:
        feather_blur_kernel_size = 15 if feather_blur_kernel_size <= 0 else feather_blur_kernel_size + 1 # Asegura que sea impar y positivo
        print(f"Advertencia: feather_blur_kernel_size ajustado a {feather_blur_kernel_size} (debe ser impar y positivo).")

    feather_blur_kernel_tuple = (feather_blur_kernel_size, feather_blur_kernel_size)
    blurred_mask = cv2.GaussianBlur(mask_inverted, feather_blur_kernel_tuple, 0)
    
    alpha_channel = np.interp(blurred_mask,
                              [alpha_threshold_bg, alpha_threshold_fg],
                              [0, 255]).astype(np.uint8)

    # 4. Combinar con la imagen original
    b, g, r = cv2.split(image_np)
    rgba_image = cv2.merge([b, g, r, alpha_channel])

    return rgba_image

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

        # Aplicar la lógica de Chroma Key
        rgba_image = _apply_chroma_key_logic(
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
    # Recibimos todos los parámetros del perfil como en /procesar-foto
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
    Recibe una imagen A4 con múltiples púas en un fondo chroma,
    elimina el fondo, detecta cada púa y devuelve un archivo ZIP
    con cada púa recortada como un archivo WebP individual.
    """
    try:
        # Leer la imagen A4
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_a4 = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Lee como BGR

        if img_a4 is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen A4.")

        # 1. Aplicar Chroma Key a toda la imagen A4
        processed_a4_img_rgba = _apply_chroma_key_logic(
            img_a4,
            lower_hsv_h, lower_hsv_s, lower_hsv_v,
            upper_hsv_h, upper_hsv_s, upper_hsv_v,
            feather_blur_kernel_size,
            alpha_threshold_fg, alpha_threshold_bg,
            morph_kernel_size, morph_iterations
        )

        # 2. Detección de Contornos (para identificar cada púa)
        # Usamos el canal alfa para encontrar los contornos de los objetos
        alpha_channel_for_contours = processed_a4_img_rgba[:, :, 3] # Canal alfa
        
        # Umbralización para obtener una máscara binaria clara de los objetos
        # Un umbral bajo (ej. 10) es efectivo si el fondo es 0 y el objeto >0
        _, binary_mask = cv2.threshold(alpha_channel_for_contours, 10, 255, cv2.THRESH_BINARY)
        
        # Operaciones morfológicas adicionales para limpiar la máscara antes de encontrar contornos
        # Esto ayuda a cerrar pequeños huecos y separar objetos ligeramente unidos
        # Asegúrate de que el kernel sea impar y positivo
        batch_morph_kernel_size = 5 # Tamaño del kernel para la detección de púas
        if batch_morph_kernel_size % 2 == 0: batch_morph_kernel_size += 1
        kernel_morph_batch = np.ones((batch_morph_kernel_size, batch_morph_kernel_size), np.uint8) 
        
        # Cerrar para unir pequeños huecos, Abrir para eliminar pequeños ruidos
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_morph_batch, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_morph_batch, iterations=2)

        # Encontrar contornos
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- AÑADIR ESTOS PRINTS TEMPORALES PARA DEPURACIÓN ---
        print("\n--- INICIO DE DEBUG: Áreas de Contornos Detectados ---")
        if not contours:
            print("No se detectaron contornos en la imagen A4. Intenta ajustar los parámetros HSV o los parámetros morfológicos.")
        # --- FIN DE PRINTS TEMPORALES ---

        output_images = []
        # --- AJUSTA ESTOS VALORES SEGÚN EL TAMAÑO DE TUS PÚAS EN LA IMAGEN A4 ---
        # Si tus púas son pequeñas en una A4 grande, reduce min_pua_area.
        # Si detecta todo el A4, reduce max_pua_area.
        MIN_PUA_AREA = 1000  # Área mínima para considerar un contorno como una púa (AJUSTA AQUÍ)
        MAX_PUA_AREA = 400000 # Área máxima para evitar detectar la hoja completa (AJUSTA AQUÍ)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # --- AÑADIR ESTE PRINT TEMPORAL ---
            print(f"Contorno {i+1}: Área = {int(area)} píxeles cuadrados")
            # --- FIN DE PRINT TEMPORAL ---
            
            # Filtrar por tamaño para evitar ruido y detectar solo objetos de interés
            if MIN_PUA_AREA < area < MAX_PUA_AREA: 
                # Obtener el rectángulo delimitador del contorno
                x, y, w, h = cv2.boundingRect(contour)

                # Expandir el bounding box ligeramente para asegurar que no se corten los bordes
                padding = 20 # Píxeles de padding. Ajusta si necesitas más o menos margen.
                
                # Calcular coordenadas con padding, asegurándose de no salirse de los límites de la imagen
                x_padded = max(0, x - padding)
                y_padded = max(0, y - padding)
                
                x2_padded = min(processed_a4_img_rgba.shape[1], x + w + padding)
                y2_padded = min(processed_a4_img_rgba.shape[0], y + h + padding)
                
                # Ancho y alto finales del recorte
                w_final = x2_padded - x_padded
                h_final = y2_padded - y_padded

                # Recortar la púa de la imagen procesada (RGBA)
                pua_cropped_rgba = processed_a4_img_rgba[y_padded:y2_padded, x_padded:x2_padded]

                if pua_cropped_rgba.size == 0: # Evitar errores si el recorte es vacío
                    continue

                # Convertir a PIL Image para guardar como WebP
                pil_pua = Image.fromarray(cv2.cvtColor(pua_cropped_rgba, cv2.COLOR_BGRA2RGBA))

                # Guardar en un buffer
                img_byte_arr = io.BytesIO()
                pil_pua.save(img_byte_arr, format="WEBP", quality=80, method=6)
                img_byte_arr.seek(0)
                output_images.append({'name': f'pua_{i+1}.webp', 'data': img_byte_arr})
        
        if not output_images:
            raise HTTPException(status_code=404, detail=f"No se detectaron púas en la imagen A4. Intenta ajustar los parámetros HSV, la separación de las púas o los valores de MIN_PUA_AREA ({MIN_PUA_AREA}) y MAX_PUA_AREA ({MAX_PUA_AREA}).")

        # 3. Comprimir todas las imágenes en un archivo ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for img_info in output_images:
                zf.writestr(img_info['name'], img_info['data'].getvalue())
        zip_buffer.seek(0)

        return StreamingResponse(zip_buffer, media_type="application/zip",
                                 headers={"Content-Disposition": "attachment; filename=puas_procesadas.zip"})

    except HTTPException: # Re-lanza HTTPExceptions ya definidas
        raise
    except Exception as e:
        print(f"ERROR en batch-process: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error interno del servidor al procesar por lotes: {e}")
