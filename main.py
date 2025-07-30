from fastapi import FastAPI, UploadFile, File, Response, HTTPException, status, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import cv2
import os

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
DEFAULT_LOWER_HSV_H = 140
DEFAULT_LOWER_HSV_S = 50
DEFAULT_LOWER_HSV_V = 50
DEFAULT_UPPER_HSV_H = 170
DEFAULT_UPPER_HSV_S = 255
DEFAULT_UPPER_HSV_V = 255

DEFAULT_FEATHER_BLUR_KERNEL_SIZE = 15 # Asegúrate de que los valores sean impares
DEFAULT_ALPHA_THRESHOLD_FG = 180
DEFAULT_ALPHA_THRESHOLD_BG = 50
DEFAULT_MORPH_KERNEL_SIZE = 5 # Asegúrate de que el valor sea impar
DEFAULT_MORPH_ITERATIONS = 2

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

# 📸 Endpoint para procesar la imagen y quitar el fondo
@app.post("/procesar-foto", summary="Process image and remove background with provided parameters")
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

        # Validaciones de kernel_size para asegurar que sean impares y positivos
        if feather_blur_kernel_size is not None and (feather_blur_kernel_size % 2 == 0 or feather_blur_kernel_size <= 0):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="feather_blur_kernel_size must be an odd, positive integer.")
        if morph_kernel_size is not None and (morph_kernel_size % 2 == 0 or morph_kernel_size <= 0):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="morph_kernel_size must be an odd, positive integer.")

        # --- APLICACIÓN DEL ALGORITMO CHROMA KEY CON PARÁMETROS DINÁMICOS ---
        
        # 1. Definir los límites HSV
        lower_hsv = np.array([
            lower_hsv_h if lower_hsv_h is not None else DEFAULT_LOWER_HSV_H,
            lower_hsv_s if lower_hsv_s is not None else DEFAULT_LOWER_HSV_S,
            lower_hsv_v if lower_hsv_v is not None else DEFAULT_LOWER_HSV_V
        ])
        upper_hsv = np.array([
            upper_hsv_h if upper_hsv_h is not None else DEFAULT_UPPER_HSV_H,
            upper_hsv_s if upper_hsv_s is not None else DEFAULT_UPPER_HSV_S,
            upper_hsv_v if upper_hsv_v is not None else DEFAULT_UPPER_HSV_V
        ])

        hsv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        mask_inverted = cv2.bitwise_not(mask)

        # 2. Operaciones morfológicas
        current_morph_kernel_size = morph_kernel_size if morph_kernel_size is not None else DEFAULT_MORPH_KERNEL_SIZE
        kernel_morph = np.ones((current_morph_kernel_size, current_morph_kernel_size), np.uint8)
            
        current_morph_iterations = morph_iterations if morph_iterations is not None else DEFAULT_MORPH_ITERATIONS
            
        mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_OPEN, kernel_morph, iterations=current_morph_iterations)
        mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_CLOSE, kernel_morph, iterations=current_morph_iterations)

        # 3. Desenfoque y canal alfa
        current_feather_blur_kernel_size = feather_blur_kernel_size if feather_blur_kernel_size is not None else DEFAULT_FEATHER_BLUR_KERNEL_SIZE
        feather_blur_kernel_tuple = (current_feather_blur_kernel_size, current_feather_blur_kernel_size)
        blurred_mask = cv2.GaussianBlur(mask_inverted, feather_blur_kernel_tuple, 0)
            
        current_alpha_threshold_bg = alpha_threshold_bg if alpha_threshold_bg is not None else DEFAULT_ALPHA_THRESHOLD_BG
        current_alpha_threshold_fg = alpha_threshold_fg if alpha_threshold_fg is not None else DEFAULT_ALPHA_THRESHOLD_FG

        alpha_channel = np.interp(blurred_mask,
                                  [current_alpha_threshold_bg, current_alpha_threshold_fg],
                                  [0, 255]).astype(np.uint8)

        # 4. Combinar con la imagen original
        b, g, r = cv2.split(opencv_image)
        rgba_image = cv2.merge([b, g, r, alpha_channel])

        # 5. Convertir a PIL Image
        output_pil_image = Image.fromarray(cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGBA))
        
        # --- NUEVO CÓDIGO: Redimensionar la imagen si su ancho es mayor a 350px ---
        MAX_WIDTH_OUTPUT = 350
        if output_pil_image.width > MAX_WIDTH_OUTPUT:
            # Calcula la nueva altura manteniendo la proporción
            aspect_ratio = output_pil_image.height / output_pil_image.width
            new_height = int(MAX_WIDTH_OUTPUT * aspect_ratio)
            # Redimensiona la imagen
            # Image.LANCZOS es un filtro de alta calidad para el downsampling
            output_pil_image = output_pil_image.resize((MAX_WIDTH_OUTPUT, new_height), Image.LANCZOS)
        # --- FIN DEL NUEVO CÓDIGO ---

        # 6. Guardar en buffer como WebP
        output_buffer = io.BytesIO()
        output_pil_image.save(output_buffer, format="WEBP", quality=80, method=6) 
        output_buffer.seek(0) 

        # 7. Devolver la respuesta con el tipo MIME correcto
        return Response(content=output_buffer.getvalue(), media_type="image/webp")

    except HTTPException: # Re-lanza HTTPExceptions ya definidas (ej. 400 Bad Request)
        raise
    except Exception as e:
        print(f"ERROR: Fallo al procesar la foto (chroma key OpenCV): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing image")
