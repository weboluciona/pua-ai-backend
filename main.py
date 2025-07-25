from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from rembg import remove, new_session

# 🚀 Inicializar FastAPI
app = FastAPI()

# 🔁 Crear sesión rembg con modelo liviano
session = new_session(model_name="u2netp")

# 📍 Endpoint base
@app.get("/")
def home():
    return {"status": "✅ Servidor de recorte de fondo activo"}

# ❤️ Endpoint de salud (para Render)
@app.get("/healthz")
def health_check():
    return {"status": "ok"}

# 📸 Endpoint principal para recortar imagen
@app.post("/procesar-foto")
async def procesar_foto(file: UploadFile = File(...)):
    # 📥 Leer imagen enviada desde el frontend (formData → "file")
    input_bytes = await file.read()

    # ✂️ Recortar fondo con rembg (usando sesión pre-cargada)
    output_bytes = remove(input_bytes, session=session)

    # 📦 Devolver imagen como PNG
    return Response(content=output_bytes, media_type="image/png")
