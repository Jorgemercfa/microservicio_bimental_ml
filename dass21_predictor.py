import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import re
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Configuración de logging MEJORADA para Azure
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler("dass21_api.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("dass21")

# Índices DASS-21 (1-based)
DEP_INDICES = [3, 5, 10, 13, 16, 17, 21]
ANS_INDICES = [2, 4, 7, 9, 15, 19, 20]
EST_INDICES = [1, 6, 8, 11, 12, 14, 18]

# -----------------------------
# NORMALIZADOR Y MAPEO DE ETIQUETAS
# -----------------------------
def normalize_text_basic(text: str):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Mapeo directo como fallback
LABEL_MAP_SPANISH = {
    "no": 0, "nunca": 0, "casi nunca": 0,
    "raras veces": 1, "ocasionalmente": 1, "algunas veces": 1, "sí, poco": 1, "si, poco": 1,
    "bastante": 2, "frecuentemente": 2, "usualmente": 2, "casi siempre": 2, "sí, bastante": 2, "si, bastante": 2,
    "siempre": 3, "sí, mucho": 3, "si, mucho": 3
}

# -----------------------------
# MODELOS DE REQUEST
# -----------------------------
class TextListRequest(BaseModel):
    respuestas: List[str]

class ChatRequest(BaseModel):
    message: str

# -----------------------------
# PREDICTOR DASS-21 OPTIMIZADO
# -----------------------------
class DASS21Predictor:
    def __init__(self, model_path: str, tokenizer_path: str, max_len: int = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer no encontrado: {tokenizer_path}")

        # Cargar modelo TFLite OPTIMIZADO
        try:
            # Fallback a tf.lite.Interpreter (más compatible con Azure)
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            logger.info("Usando tf.lite.Interpreter (optimizado para Azure)")
            
            self.interpreter.allocate_tensors()
        except Exception as e:
            logger.error(f"Error cargando modelo TFLite: {e}")
            raise

        # Obtener parámetros del modelo
        input_details = self.interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        self.max_len = max_len or (input_shape[1] if len(input_shape) >= 2 else 100)
        self.input_dtype = input_details[0]['dtype']
        
        logger.info(f"Modelo cargado. Input shape: {input_shape}, dtype: {self.input_dtype}")

        # Cargar tokenizer - VERSIÓN SIMPLIFICADA
        try:
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                tokenizer_content = f.read().strip()
            
            self.tokenizer = tokenizer_from_json(tokenizer_content)
            logger.info(f"Tokenizer cargado con vocab_size={len(self.tokenizer.word_index)}")
            
        except Exception as e:
            logger.error(f"Error cargando tokenizer: {e}")
            # Crear tokenizer básico como fallback
            from tensorflow.keras.preprocessing.text import Tokenizer
            self.tokenizer = Tokenizer(oov_token="<OOV>")
            logger.warning("Usando tokenizer básico como fallback")

        logger.info(f"Longitud máxima: {self.max_len}")

    def tokenize_input(self, input_text: str):
        text = normalize_text_basic(input_text)
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
        
        # Convertir al dtype esperado por el modelo
        return np.array(padded, dtype=self.input_dtype)

    def _predict_question(self, text: str) -> int:
        # Fallback directo si está en el mapeo (PRIMERO)
        key = normalize_text_basic(text)
        if key in LABEL_MAP_SPANISH:
            logger.info(f"Usando mapeo directo para: '{text}' -> {LABEL_MAP_SPANISH[key]}")
            return LABEL_MAP_SPANISH[key]

        # Si no está en el mapeo, usar el modelo
        try:
            inp = self.tokenize_input(text)
            input_index = self.interpreter.get_input_details()[0]['index']
            self.interpreter.set_tensor(input_index, inp)
            self.interpreter.invoke()
            out_index = self.interpreter.get_output_details()[0]['index']
            out = self.interpreter.get_tensor(out_index)
            prediction = int(np.argmax(out, axis=1)[0])
            logger.info(f"Predicción modelo para: '{text}' -> {prediction}")
            return prediction
        except Exception as e:
            logger.warning(f"Error en predicción del modelo, usando fallback: {e}")
            return LABEL_MAP_SPANISH.get(key, 1)  # Default a 1

    def predict_scores_from_text_list(self, respuestas: List[str]) -> Dict[str, int]:
        if not isinstance(respuestas, list) or len(respuestas) != 21:
            raise ValueError("Se requieren exactamente 21 respuestas para el cuestionario DASS-21.")
        
        per_question_vals = [self._predict_question(r) for r in respuestas]

        dep_total = sum(per_question_vals[i-1] for i in DEP_INDICES)
        ans_total = sum(per_question_vals[i-1] for i in ANS_INDICES)
        est_total = sum(per_question_vals[i-1] for i in EST_INDICES)

        return {
            "depresion": dep_total,
            "ansiedad": ans_total,
            "estres": est_total,
            "respuestas": per_question_vals
        }

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="DASS-21 API", version="1.0")

# CORS configurado para Azure
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Buscar automáticamente modelo y tokenizer - VERSIÓN ROBUSTA
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.warning(f"Directorio 'model' creado en: {MODEL_DIR}")

# Búsqueda flexible de archivos
def find_model_files():
    tflite_files = []
    json_files = []
    
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".tflite"):
                tflite_files.append(f)
            elif f.endswith(".json") and "tokenizer" in f.lower():
                json_files.append(f)
    
    return tflite_files, json_files

tflite_files, json_files = find_model_files()

if not tflite_files:
    logger.error(f"No se encontraron archivos .tflite en {MODEL_DIR}")
    MODEL_PATH = None
else:
    MODEL_PATH = os.path.join(MODEL_DIR, tflite_files[0])
    logger.info(f"Usando modelo: {MODEL_PATH}")

if not json_files:
    logger.error(f"No se encontraron archivos tokenizer .json en {MODEL_DIR}")
    TOKENIZER_PATH = None
else:
    TOKENIZER_PATH = os.path.join(MODEL_DIR, json_files[0])
    logger.info(f"Usando tokenizer: {TOKENIZER_PATH}")

predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    try:
        if MODEL_PATH and TOKENIZER_PATH:
            predictor = DASS21Predictor(MODEL_PATH, TOKENIZER_PATH)
            logger.info("✅ Modelo y tokenizer cargados exitosamente.")
            
            # Probar una predicción simple
            test_result = predictor._predict_question("no")
            logger.info(f"✅ Prueba de predicción exitosa: 'no' -> {test_result}")
        else:
            raise FileNotFoundError("No se encontraron archivos de modelo o tokenizer")
        
    except Exception as e:
        logger.exception("❌ Error cargando el modelo/tokenizer, usando alternativa")
        predictor = AlternativeDASS21Predictor()
        logger.info("✅ Predictor alternativo inicializado")

# -----------------------------
# PREDICTOR ALTERNATIVO MEJORADO
# -----------------------------
class AlternativeDASS21Predictor:
    """Predictor alternativo que usa solo el mapeo de etiquetas"""
    def __init__(self):
        logger.info("Usando predictor alternativo (solo mapeo de etiquetas)")
        
    def _predict_question(self, text: str) -> int:
        key = normalize_text_basic(text)
        return LABEL_MAP_SPANISH.get(key, 1)  # Default a 1 si no se encuentra
    
    def predict_scores_from_text_list(self, respuestas: List[str]) -> Dict[str, int]:
        if not isinstance(respuestas, list) or len(respuestas) != 21:
            raise ValueError("Se requieren exactamente 21 respuestas.")
        
        per_question_vals = [self._predict_question(r) for r in respuestas]
        
        dep_total = sum(per_question_vals[i-1] for i in DEP_INDICES)
        ans_total = sum(per_question_vals[i-1] for i in ANS_INDICES)
        est_total = sum(per_question_vals[i-1] for i in EST_INDICES)

        return {
            "depresion": dep_total,
            "ansiedad": ans_total,
            "estres": est_total,
            "respuestas": per_question_vals,
            "nota": "Usando mapeo directo (modelo no disponible)"
        }

# -----------------------------
# ENDPOINTS
# -----------------------------
@app.post("/predict/")
def predict_scores(request: TextListRequest):
    if not request.respuestas or not isinstance(request.respuestas, list):
        raise HTTPException(status_code=400, detail="Se requiere una lista de respuestas (no vacía)")
    if len(request.respuestas) != 21:
        raise HTTPException(status_code=422, detail="Debe enviar exactamente 21 respuestas para el DASS-21.")
    try:
        result = predictor.predict_scores_from_text_list(request.respuestas)
        return result
    except Exception as e:
        logger.exception("Error al predecir los puntajes")
        raise HTTPException(status_code=500, detail=f"Error al predecir los puntajes: {str(e)}")

# -----------------------------
# FUNCIÓN PARA OPENAI - MEJORADA
# -----------------------------
def consultar_openai(mensaje: str) -> str:
    # Verificar si la API Key está configurada
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY no configurada en variables de entorno")
        return "Lo siento, el servicio de chat no está configurado correctamente. Por favor, contacta al administrador del sistema."
    
    try:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Eres un asistente empático especializado en salud mental. Responde de forma breve, clara y respetuosa."},
                {"role": "user", "content": mensaje}
            ],
            "max_tokens": 200,
            "temperature": 0.7,
        }
        
        logger.info(f"Enviando consulta a OpenAI: {mensaje[:50]}...")
        
        response = requests.post(
            OPENAI_API_URL,
            headers={
                "Content-Type": "application/json", 
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            respuesta = data["choices"][0]["message"]["content"].strip()
            logger.info("✅ Respuesta de OpenAI recibida correctamente")
            return respuesta
        else:
            logger.error(f"❌ Error OpenAI API: {response.status_code} - {response.text}")
            return "Lo siento, no puedo responder en este momento. Por favor intenta más tarde."
            
    except requests.exceptions.Timeout:
        logger.error("⏰ Timeout al consultar OpenAI")
        return "El servicio está tardando demasiado en responder. Por favor intenta nuevamente."
    except Exception as e:
        logger.error(f"❌ Error consultando OpenAI: {e}")
        return "Lo siento, hay un problema temporal con el servicio. Por favor intenta más tarde."

@app.post("/chat/")
def chat_with_user(request: ChatRequest):
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")
        
        logger.info(f"📨 Mensaje recibido en /chat: {user_message}")
        respuesta = consultar_openai(user_message)
        
        return {"respuesta": respuesta}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Error en el endpoint de chat")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# Endpoint de salud MEJORADO para Azure
@app.get("/health")
def health_check():
    status = "ok" if predictor is not None else "error"
    message = "API DASS-21 funcionando correctamente" if predictor else "Predictor no inicializado"
    
    # Verificar también la API Key de OpenAI
    openai_status = "configured" if OPENAI_API_KEY else "not configured"
    
    return {
        "status": status, 
        "message": message,
        "version": "1.0",
        "predictor_type": predictor.__class__.__name__ if predictor else "None",
        "openai_api_key": openai_status,
        "endpoints_available": ["/predict", "/chat", "/model-info"]
    }

# Endpoint de información del modelo
@app.get("/model-info")
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor no inicializado")
    
    model_info = {
        "model_loaded": True,
        "max_length": predictor.max_len if hasattr(predictor, 'max_len') else None,
        "input_dtype": str(predictor.input_dtype) if hasattr(predictor, 'input_dtype') else None,
        "predictor_type": predictor.__class__.__name__,
        "openai_configured": bool(OPENAI_API_KEY)
    }
    return model_info

# Endpoint raíz
@app.get("/")
def read_root():
    return {
        "message": "API DASS-21 Predictor", 
        "version": "1.0",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Estado del servicio"},
            {"path": "/predict", "method": "POST", "description": "Evaluación DASS-21"},
            {"path": "/chat", "method": "POST", "description": "Chat de apoyo emocional"},
            {"path": "/model-info", "method": "GET", "description": "Información del modelo"}
        ],
        "status": "active"
    }

# === main ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)