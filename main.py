from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from rembg import remove, new_session
import uvicorn

# 🚀 Inicializar FastAPI
app = FastAPI()

# 🔁 Crear sesión rembg (mantiene el modelo cargado en memoria)
session = new_session()

@app.get("/")
def home():
    return {"status": "✅ Servidor de recorte de púa activo"}

@app.post("/procesar-foto")
async def procesar_foto(file: UploadFile = File(...)):
    # 📥 Leer imagen enviada desde el frontend (formData → "file")
    input_bytes = await file.read()

    # ✂️ Recortar fondo con rembg (usando sesión pre-cargada)
    output_bytes = remove(input_bytes, session=session)

    # 📦 Devolver imagen como PNG
    return Response(content=output_bytes, media_type="image/png")

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
