from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from rembg import remove, new_session
import uvicorn

# 🚀 Crear app FastAPI
app = FastAPI()

# 🔁 Crear sesión de rembg 1 sola vez (esto ahorra RAM en Render)
session = new_session()

@app.get("/")
def home():
    return {"status": "🪙 Motor IA de Recorte de Púa activo"}

@app.post("/procesar-foto")
async def procesar_foto(file: UploadFile = File(...)):
    # 📥 Leer la imagen enviada
    input_bytes = await file.read()

    # ✨ Eliminar fondo con sesión precargada
    output_bytes = remove(input_bytes, session=session)

    # 📦 Devolver imagen ya recortada (PNG por defecto)
    return Response(content=output_bytes, media_type="image/png")

# 🛠️ Para ejecutar localmente (no necesario en Render)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
