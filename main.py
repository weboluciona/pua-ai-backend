from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image # Todavía útil para abrir/guardar y trabajar con IO
import io
import numpy as np
import cv2 # ¡Importamos OpenCV!

# 🚀 Inicializar FastAPI
app = FastAPI()

# 🛡️ Añadir CORS para que tu HTML pueda hacer peticiones sin bloqueo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Puedes restringirlo a tu dominio si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🎨 PARAMETROS CLAVE PARA AJUSTAR LA ELIMINACIÓN DEL FONDO (¡LEE Y AJUSTA!)
# Para tu fondo rosa, es mejor trabajar en el espacio de color HSV.
# Hue (Tono), Saturation (Saturación), Value (Brillo).
# Los valores H van de 0-179 en OpenCV (no 0-360 como en otros sistemas).
# Los valores S y V van de 0-255.

# **AJUSTA ESTOS LÍMITES PARA EL ROSA/MAGENTA DE TU FONDO**
# Puedes usar una herramienta de selección de color para tu imagen (ej. en un editor de fotos)
# y luego convertir RGB a HSV (hay conversores online).
# Ejemplo para un rosa/magenta vibrante:
# R=255, G=0, B=255 (magenta puro) -> H=150, S=255, V=255 (aproximado en OpenCV)
# Hemos de dar un rango. Si el rosa tiene algo de rojo, su H será menor.
# Si es más tirando a morado, su H será mayor.
# Esto es un punto de partida para tu imagen:
LOWER_HSV_BOUND = np.array([140, 50, 50]) # Tono bajo, Saturación baja, Brillo bajo
UPPER_HSV_BOUND = np.array([170, 255, 255]) # Tono alto, Saturación alta, Brillo alto

# **PARÁMETROS PARA EL RECORTE Y DIFUMINADO (FEATHERING)**
FEATHER_BLUR_KERNEL = (15, 15) # Tamaño del kernel para el desenfoque del borde (feathering).
                               # Números impares y más grandes = más difuminado. (ej: (5,5), (25,25))
ALPHA_THRESHOLD_FG = 180     # Umbral de la máscara para el primer plano (foreground).
                               # Píxeles con valor alfa >= a este se considerarán opacos.
                               # Ajusta para hacer el corte más nítido o más suave. (0-255)
ALPHA_THRESHOLD_BG = 50      # Umbral de la máscara para el fondo (background).
                               # Píxeles con valor alfa <= a este se considerarán transparentes.
                               # Ajusta para afinar la eliminación de restos de fondo. (0-255)

# **PARÁMETROS PARA LIMPIAR LA MÁSCARA (opcional, para ruido o huecos)**
MORPH_KERNEL_SIZE = 5         # Tamaño del kernel para operaciones morfológicas (ej: 3, 5, 7)
                               # Más grande = más efecto.
ITERATIONS = 2                # Número de iteraciones para operaciones morfológicas.


# 🩺 Endpoint de salud para Render
@app.get("/healthz")
def health_check():
    return {"status": "ok"}

# 🏠 Endpoint base por si visitas la raíz en el navegador
@app.get("/")
def home():
    return {"status": "✅ Servidor de recorte de fondo activo (método chroma key avanzado)"}

# 📸 Endpoint para procesar la imagen y quitar el fondo
@app.post("/procesar-foto")
async def procesar_foto(file: UploadFile = File(...)):
    try:
        # 📥 Leer imagen enviada
        input_bytes = await file.read()

        # Convertir bytes a imagen de Pillow, luego a array de OpenCV (BGR)
        pil_image = Image.open(io.BytesIO(input_bytes)).convert("RGB") # Leer como RGB
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) # Convertir a BGR (formato de OpenCV)

        # 1. Convertir la imagen a HSV para una mejor detección de color
        hsv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)

        # 2. Crear una máscara para el color del fondo
        # Los píxeles dentro del rango definido serán blancos (255), el resto negros (0).
        mask = cv2.inRange(hsv_image, LOWER_HSV_BOUND, UPPER_HSV_BOUND)

        # 3. Invertir la máscara: Ahora el objeto de la púa es blanco (255) y el fondo es negro (0).
        mask_inverted = cv2.bitwise_not(mask)

        # 4. (Opcional) Limpiar la máscara con operaciones morfológicas
        # 'Opening' elimina pequeños puntos blancos (ruido) fuera del objeto.
        # 'Closing' rellena pequeños agujeros dentro del objeto.
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_OPEN, kernel, iterations=ITERATIONS)
        mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_CLOSE, kernel, iterations=ITERATIONS)

        # 5. Aplicar desenfoque (Gaussian Blur) a la máscara para el feathering (difuminado)
        # Esto suaviza los bordes de la máscara.
        blurred_mask = cv2.GaussianBlur(mask_inverted, FEATHER_BLUR_KERNEL, 0)

        # 6. Escalar la máscara borrosa para el canal alfa
        # Donde la máscara borrosa es > ALPHA_THRESHOLD_FG, el alfa es opaco.
        # Donde la máscara borrosa es < ALPHA_THRESHOLD_BG, el alfa es transparente.
        # Entre esos valores, habrá semitransparencia (feathering).
        alpha_channel = np.interp(blurred_mask, [ALPHA_THRESHOLD_BG, ALPHA_THRESHOLD_FG], [0, 255]).astype(np.uint8)

        # 7. Combinar los canales RGB de la imagen original con el nuevo canal alfa
        b, g, r = cv2.split(opencv_image)
        rgba_image = cv2.merge([b, g, r, alpha_channel])

        # Convertir de OpenCV (NumPy array) a Pillow Image para guardar en bytes
        output_pil_image = Image.fromarray(cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGBA))

        # 📤 Guardar la imagen en un buffer de bytes como PNG (necesario para la transparencia)
        output_buffer = io.BytesIO()
        output_pil_image.save(output_buffer, format="PNG")
        output_bytes = output_buffer.getvalue()

        # 📤 Devolver imagen como PNG
        return Response(content=output_bytes, media_type="image/png")

    except Exception as e:
        # ❌ Manejo de errores
        print(f"ERROR: Fallo al procesar la foto (chroma key OpenCV): {e}")
        return Response(content=str(e), media_type="text/plain", status_code=500)
