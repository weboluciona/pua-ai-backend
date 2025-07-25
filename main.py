from flask import Flask, request, send_file
from rembg import remove
from PIL import Image
import io
import time

app = Flask(__name__)

@app.route('/')
def home():
    return '🪙 Motor IA de Recorte de Púa funcionando'

@app.route('/procesar-foto', methods=['POST'])
def procesar_foto():
    if 'imagen' not in request.files:
        return 'No se recibió imagen', 400

    file = request.files['imagen']
    input_image = Image.open(file.stream).convert("RGBA")
    output_image = remove(input_image)

    output_io = io.BytesIO()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"pua_movil_{timestamp}.webp"
    output_image.save(output_io, format="WEBP")
    output_io.seek(0)

    return send_file(output_io, mimetype='image/webp',
                     as_attachment=True, download_name=nombre_archivo)

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render te asigna el puerto
    app.run(host='0.0.0.0', port=port)
