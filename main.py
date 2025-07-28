from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session

# 🚀 Inicializar FastAPI
app = FastAPI()

# 🛡️ Añadir CORS para que tu HTML pueda hacer peticiones sin bloqueo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringirlo a tu dominio si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔁 Crear sesión rembg con modelo liviano (ideal para Render Free)
session = new_session(model_name="u2netp")

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

        # ✂️ Quitar fondo con rembg usando sesión pre-cargada
        output_bytes = remove(input_bytes, session=session)

        # 📤 Devolver imagen como PNG
        return Response(content=output_bytes, media_type="image/png")

    except Exception as e:
        # ❌ Manejo de errores para evitar fallos silenciosos
        return Response(content=str(e), media_type="text/plain", status_code=500)
