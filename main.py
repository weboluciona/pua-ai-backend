from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
import threading # Necesitaremos esto para asegurar una sola carga

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

# 🌐 Variable global para la sesión de rembg y un "lock" para hilos
# Esto evita que múltiples peticiones intenten cargar el modelo al mismo tiempo
rembg_session = None
session_lock = threading.Lock()

# 💡 Función para cargar la sesión de rembg de forma perezosa
def get_rembg_session():
    global rembg_session
    # Usamos un lock para que solo un hilo intente cargar la sesión a la vez
    with session_lock:
        if rembg_session is None:
            print("INFO: Cargando el modelo u2netp de rembg por primera vez...")
            rembg_session = new_session(model_name="u2netp")
            print("INFO: Modelo u2netp de rembg cargado exitosamente.")
        return rembg_session

# 🩺 Endpoint de salud para Render
@app.get("/healthz")
def health_check():
    return {"status": "ok"}

# 🏠 Endpoint base por si visitas la raíz en el navegador
@app.get("/")
def home():
    return {"status": "✅ Servidor de recorte de fondo activo"}

# 📸 Endpoint para procesar la imagen y quitar el fondo
@app.post("/procesar-foto")
async def procesar_foto(file: UploadFile = File(...)):
    try:
        # 📥 Leer imagen enviada
        input_bytes = await file.read()

        # ✂️ Obtener la sesión de rembg (se carga si no lo está ya)
        session = get_rembg_session()
        output_bytes = remove(input_bytes, session=session)

        # 📤 Devolver imagen como PNG
        return Response(content=output_bytes, media_type="image/png")

    except Exception as e:
        # ❌ Manejo de errores para evitar fallos silenciosos
        print(f"ERROR: Fallo al procesar la foto: {e}") # Añadir log de error en el servidor
        return Response(content=str(e), media_type="text/plain", status_code=500)
