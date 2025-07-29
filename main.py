from fastapi import FastAPI, UploadFile, File, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import cv2
import os
import pymysql
import pymysql.cursors
from pydantic import BaseModel, Field

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

# --- CONFIGURACIÓN DE LA BASE DE DATOS ---
# Las variables de entorno se obtendrán de la configuración de Render
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "hosting163236eu_puas_chroma")
DB_PORT = int(os.getenv("DB_PORT", 3306))

def get_db_connection():
    """
    Establece y devuelve una conexión a la base de datos MySQL.
    Utiliza DictCursor para que los resultados de las consultas sean diccionarios.
    """
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except pymysql.Error as e:
        print(f"ERROR: Fallo al conectar a la base de datos: {e}")
        # Eleva una excepción HTTP para que el cliente reciba un error 500
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to connect to database")

# --- FIN CONFIGURACIÓN DE LA BASE DE DATOS ---


# --- MODELOS DE DATOS PARA LOS PERFILES DE CHROMA (Pydantic) ---

# Modelo base para la creación y actualización de perfiles
# Incluye validaciones básicas para los rangos de valores HSV y de kernels
class ChromaProfileBase(BaseModel):
    profile_name: str = Field(..., max_length=255, description="Unique name for the chroma profile")
    description: str | None = Field(None, description="Optional description for the profile")
    lower_hsv_h: int | None = Field(None, ge=0, le=179, description="Lower Hue bound (0-179)")
    lower_hsv_s: int | None = Field(None, ge=0, le=255, description="Lower Saturation bound (0-255)")
    lower_hsv_v: int | None = Field(None, ge=0, le=255, description="Lower Value bound (0-255)")
    upper_hsv_h: int | None = Field(None, ge=0, le=179, description="Upper Hue bound (0-179)")
    upper_hsv_s: int | None = Field(None, ge=0, le=255, description="Upper Saturation bound (0-255)")
    upper_hsv_v: int | None = Field(None, ge=0, le=255, description="Upper Value bound (0-255)")
    feather_blur_kernel_size: int | None = Field(None, ge=1, le=101, description="Kernel size for Gaussian blur (must be odd, e.g., 5, 15, 25)")
    alpha_threshold_fg: int | None = Field(None, ge=0, le=255, description="Alpha threshold for foreground pixels (0-255)")
    alpha_threshold_bg: int | None = Field(None, ge=0, le=255, description="Alpha threshold for background pixels (0-255)")
    morph_kernel_size: int | None = Field(None, ge=1, le=101, description="Kernel size for morphological operations (must be odd, e.g., 3, 5, 7)")
    morph_iterations: int | None = Field(None, ge=0, description="Number of iterations for morphological operations")

# Modelo para la creación de un nuevo perfil (profile_name es obligatorio al crear)
class ChromaProfileCreate(ChromaProfileBase):
    pass # Hereda todos los campos de ChromaProfileBase, donde profile_name ya es obligatorio

# Modelo para leer un perfil desde la base de datos (incluye el 'id' generado por la DB)
class ChromaProfile(ChromaProfileBase):
    id: int # El ID es obligatorio al leer un perfil existente

    class Config:
        from_attributes = True # Permite crear el modelo directamente desde un objeto o diccionario de la DB

# --- FIN MODELOS DE DATOS ---


# 🎨 PARAMETROS CLAVE PARA AJUSTAR LA ELIMINACIÓN DEL FONDO (Estos están hardcodeados POR AHORA)
# En el siguiente paso, estos valores se cargarán dinámicamente desde la base de datos.
LOWER_HSV_BOUND = np.array([140, 50, 50])
UPPER_HSV_BOUND = np.array([170, 255, 255])

FEATHER_BLUR_KERNEL = (15, 15) # Asegúrate de que los valores sean impares y positivos
ALPHA_THRESHOLD_FG = 180
ALPHA_THRESHOLD_BG = 50
MORPH_KERNEL_SIZE = 5 # Asegúrate de que el valor sea impar y positivo
ITERATIONS = 2


# 🩺 Endpoint de salud para Render
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

# --- ENDPOINTS CRUD PARA chroma_settings_profiles ---

@app.post("/profiles/", response_model=ChromaProfile, status_code=status.HTTP_201_CREATED, summary="Create a new Chroma Profile")
async def create_profile(profile: ChromaProfileCreate):
    """
    Crea un nuevo perfil de configuración para la eliminación de chroma key.
    Requiere un `profile_name` único y puede incluir todos los parámetros de ajuste.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Comprobar si el profile_name ya existe para evitar duplicados
                cursor.execute("SELECT id FROM chroma_settings_profiles WHERE profile_name = %s", (profile.profile_name,))
                if cursor.fetchone():
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Profile with this name already exists")

                query = """
                INSERT INTO chroma_settings_profiles (
                    profile_name, description,
                    lower_hsv_h, lower_hsv_s, lower_hsv_v,
                    upper_hsv_h, upper_hsv_s, upper_hsv_v,
                    feather_blur_kernel_size, alpha_threshold_fg, alpha_threshold_bg,
                    morph_kernel_size, morph_iterations
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    profile.profile_name, profile.description,
                    profile.lower_hsv_h, profile.lower_hsv_s, profile.lower_hsv_v,
                    profile.upper_hsv_h, profile.upper_hsv_s, profile.upper_hsv_v,
                    profile.feather_blur_kernel_size, profile.alpha_threshold_fg, profile.alpha_threshold_bg,
                    profile.morph_kernel_size, profile.morph_iterations
                ))
                conn.commit()
                profile_id = cursor.lastrowid # Obtener el ID del nuevo registro auto-incrementado

                # Devolver el perfil creado, incluyendo su ID
                return ChromaProfile(id=profile_id, **profile.model_dump())

    except HTTPException: # Re-lanza HTTPExceptions ya definidas (ej. 409 Conflict)
        raise
    except pymysql.Error as e:
        print(f"ERROR: Fallo de DB al crear perfil: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error creating profile")
    except Exception as e:
        print(f"ERROR: Error desconocido al crear perfil: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@app.get("/profiles/", response_model=list[ChromaProfile], summary="Get all Chroma Profiles")
async def get_all_profiles():
    """
    Recupera una lista de todos los perfiles de configuración de chroma key almacenados.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM chroma_settings_profiles ORDER BY profile_name")
                profiles = cursor.fetchall()
                # Convierte cada diccionario de resultado en una instancia del modelo ChromaProfile
                return [ChromaProfile(**profile) for profile in profiles]
    except pymysql.Error as e:
        print(f"ERROR: Fallo de DB al obtener perfiles: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error fetching profiles")
    except Exception as e:
        print(f"ERROR: Error desconocido al obtener perfiles: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@app.get("/profiles/{profile_id}", response_model=ChromaProfile, summary="Get a Chroma Profile by ID")
async def get_profile(profile_id: int):
    """
    Recupera un perfil de configuración de chroma key específico por su ID.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM chroma_settings_profiles WHERE id = %s", (profile_id,))
                profile = cursor.fetchone() # fetchone() porque esperamos un solo resultado
                if not profile:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
                return ChromaProfile(**profile)
    except HTTPException: # Re-lanza HTTPExceptions ya definidas (ej. 404 Not Found)
        raise
    except pymysql.Error as e:
        print(f"ERROR: Fallo de DB al obtener perfil por ID: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error fetching profile")
    except Exception as e:
        print(f"ERROR: Error desconocido al obtener perfil por ID: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@app.put("/profiles/{profile_id}", response_model=ChromaProfile, summary="Update a Chroma Profile by ID")
async def update_profile(profile_id: int, profile: ChromaProfileBase):
    """
    Actualiza un perfil de configuración de chroma key existente.
    Todos los campos se pueden actualizar. El `profile_name` debe seguir siendo único.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # 1. Comprobar si el perfil existe
                cursor.execute("SELECT id FROM chroma_settings_profiles WHERE id = %s", (profile_id,))
                if not cursor.fetchone():
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

                # 2. Comprobar si el nuevo profile_name ya existe para otro ID (evitar duplicados)
                if profile.profile_name: # Solo si se proporciona un profile_name en la actualización
                     cursor.execute("SELECT id FROM chroma_settings_profiles WHERE profile_name = %s AND id != %s", (profile.profile_name, profile_id))
                     if cursor.fetchone():
                         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Profile with this name already exists for another ID")

                query = """
                UPDATE chroma_settings_profiles SET
                    profile_name = %s, description = %s,
                    lower_hsv_h = %s, lower_hsv_s = %s, lower_hsv_v = %s,
                    upper_hsv_h = %s, upper_hsv_s = %s, upper_hsv_v = %s,
                    feather_blur_kernel_size = %s, alpha_threshold_fg = %s, alpha_threshold_bg = %s,
                    morph_kernel_size = %s, morph_iterations = %s
                WHERE id = %s
                """
                cursor.execute(query, (
                    profile.profile_name, profile.description,
                    profile.lower_hsv_h, profile.lower_hsv_s, profile.lower_hsv_v,
                    profile.upper_hsv_h, profile.upper_hsv_s, profile.upper_hsv_v,
                    profile.feather_blur_kernel_size, profile.alpha_threshold_fg, profile.alpha_threshold_bg,
                    profile.morph_kernel_size, profile.morph_iterations,
                    profile_id # El ID del perfil a actualizar
                ))
                conn.commit()

                # Recuperar y devolver el perfil actualizado
                cursor.execute("SELECT * FROM chroma_settings_profiles WHERE id = %s", (profile_id,))
                updated_profile = cursor.fetchone()
                return ChromaProfile(**updated_profile)

    except HTTPException:
        raise
    except pymysql.Error as e:
        print(f"ERROR: Fallo de DB al actualizar perfil: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error updating profile")
    except Exception as e:
        print(f"ERROR: Error desconocido al actualizar perfil: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@app.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a Chroma Profile by ID")
async def delete_profile(profile_id: int):
    """
    Elimina un perfil de configuración de chroma key existente por su ID.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM chroma_settings_profiles WHERE id = %s", (profile_id,))
                conn.commit()
                if cursor.rowcount == 0: # Si no se eliminó ninguna fila, significa que el perfil no existía
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
        # Un Response con 204 No Content es estándar para eliminaciones exitosas sin cuerpo de respuesta
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except pymysql.Error as e:
        print(f"ERROR: Fallo de DB al eliminar perfil: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error deleting profile")
    except Exception as e:
        print(f"ERROR: Error desconocido al eliminar perfil: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

# 📸 Endpoint para procesar la imagen y quitar el fondo
# ESTE ENDPOINT AÚN USA LOS PARÁMETROS HARDCODEADOS.
# Lo modificaremos en el Paso 3 para que use los parámetros de la DB.
@app.post("/procesar-foto", summary="Process image and remove background")
async def procesar_foto(file: UploadFile = File(...)):
    """
    Recibe una imagen, aplica el algoritmo de chroma key con los parámetros predefinidos,
    y devuelve la imagen con el fondo eliminado (PNG con transparencia).
    """
    try:
        input_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        hsv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, LOWER_HSV_BOUND, UPPER_HSV_BOUND)
        mask_inverted = cv2.bitwise_not(mask)

        kernel_morph = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_OPEN, kernel_morph, iterations=ITERATIONS)
        mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_CLOSE, kernel_morph, iterations=ITERATIONS)

        # Aquí FEATHER_BLUR_KERNEL se asegura de que el tuple sea de enteros impares para cv2.GaussianBlur
        # Si FEATHER_BLUR_KERNEL es (15, 15), es un buen valor inicial.
        blurred_mask = cv2.GaussianBlur(mask_inverted, FEATHER_BLUR_KERNEL, 0)
        alpha_channel = np.interp(blurred_mask, [ALPHA_THRESHOLD_BG, ALPHA_THRESHOLD_FG], [0, 255]).astype(np.uint8)

        b, g, r = cv2.split(opencv_image)
        rgba_image = cv2.merge([b, g, r, alpha_channel])

        output_pil_image = Image.fromarray(cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGBA))
        output_buffer = io.BytesIO()
        output_pil_image.save(output_buffer, format="PNG")
        output_bytes = output_buffer.getvalue()

        return Response(content=output_bytes, media_type="image/png")

    except Exception as e:
        print(f"ERROR: Fallo al procesar la foto (chroma key OpenCV): {e}")
        # En un entorno de producción, evita exponer detalles internos del error.
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing image")
