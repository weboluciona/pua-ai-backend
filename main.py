from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

# 🛡️ Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto por el dominio de tu app si lo prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/procesar-foto")
async def procesar_foto(file: UploadFile = File(...)):
    try:
        # 📥 Cargar la imagen recibida
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 📐 Redimensionar si es necesario (por ejemplo, máximo 1024px de ancho)
        max_width = 1024
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height))

        # 🖼️ Comprimir a formato WEBP
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=70)
        buffer.seek(0)

        # 📤 Devolver la imagen procesada
        return Response(content=buffer.read(), media_type="image/webp")

    except Exception as e:
        return {"error": str(e)}
