import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import re
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict
import numpy as np

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dass21")

# Índices DASS-21
DEP_INDICES = [3, 5, 10, 13, 16, 17, 21]
ANS_INDICES = [2, 4, 7, 9, 15, 19, 20]
EST_INDICES = [1, 6, 8, 11, 12, 14, 18]

# Mapeo directo - SISTEMA CONFIABLE
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
# PREDICTOR MEJORADO - SIN TENSORFLOW
# -----------------------------
class DASS21Predictor:
    def __init__(self):
        logger.info("✅ Predictor DASS-21 inicializado (sistema confiable)")
        
    def normalize_text(self, text: str):
        if not isinstance(text, str):
            text = str(text)
        return text.lower().strip()
    
    def predict_question(self, text: str) -> int:
        key = self.normalize_text(text)
        prediction = LABEL_MAP_SPANISH.get(key, 1)  # Default a 1
        logger.info(f"Predicción: '{text}' -> {prediction}")
        return prediction
    
    def predict_scores(self, respuestas: List[str]) -> Dict[str, int]:
        if len(respuestas) != 21:
            raise ValueError("Se requieren exactamente 21 respuestas")
        
        valores = [self.predict_question(r) for r in respuestas]
        
        dep_total = sum(valores[i-1] for i in DEP_INDICES)
        ans_total = sum(valores[i-1] for i in ANS_INDICES)
        est_total = sum(valores[i-1] for i in EST_INDICES)

        return {
            "depresion": dep_total,
            "ansiedad": ans_total,
            "estres": est_total,
            "respuestas": valores,
            "nota": "Sistema confiable - Mapeo directo"
        }

# -----------------------------
# LIFESPAN MODERNO - SIN WARNINGS
# -----------------------------
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global predictor
    predictor = DASS21Predictor()
    logger.info("🚀 API DASS-21 iniciada correctamente")
    yield
    # Shutdown
    logger.info("🛑 API DASS-21 detenida")

# -----------------------------
# FASTAPI APP CON CORS MEJORADO
# -----------------------------
app = FastAPI(title="DASS-21 API", version="2.0", lifespan=lifespan)

# CORS MEJORADO - SOLUCIÓN DEFINITIVA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)

# -----------------------------
# ENDPOINTS MEJORADOS
# -----------------------------
@app.post("/predict/")
async def predict_scores(request: TextListRequest):
    logger.info(f"📊 Recibida solicitud /predict con {len(request.respuestas)} respuestas")
    
    if len(request.respuestas) != 21:
        raise HTTPException(422, "Debe enviar exactamente 21 respuestas")
    
    try:
        result = predictor.predict_scores(request.respuestas)
        logger.info(f"✅ Predicción exitosa: D={result['depresion']}, A={result['ansiedad']}, E={result['estres']}")
        return result
    except Exception as e:
        logger.error(f"❌ Error en /predict: {e}")
        raise HTTPException(500, f"Error al procesar: {str(e)}")

def consultar_openai(mensaje: str) -> str:
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY no configurada")
        return "Servicio de chat no configurado. Contacte al administrador."
    
    try:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Eres un asistente empático especializado en salud mental."},
                {"role": "user", "content": mensaje}
            ],
            "max_tokens": 200,
            "temperature": 0.7,
        }
        
        logger.info(f"🤖 Consultando OpenAI: {mensaje[:50]}...")
        
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
            respuesta = response.json()["choices"][0]["message"]["content"].strip()
            logger.info("✅ Respuesta OpenAI recibida")
            return respuesta
        else:
            logger.error(f"❌ Error OpenAI: {response.status_code}")
            return "Lo siento, no puedo responder en este momento."
            
    except Exception as e:
        logger.error(f"❌ Error consultando OpenAI: {e}")
        return "Error temporal del servicio."

@app.post("/chat/")
async def chat_with_user(request: ChatRequest):
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(400, "El mensaje no puede estar vacío")
        
        logger.info(f"💬 Mensaje recibido: {user_message}")
        respuesta = consultar_openai(user_message)
        logger.info(f"💭 Respuesta enviada: {respuesta[:50]}...")
        
        return {"respuesta": respuesta}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en /chat: {e}")
        raise HTTPException(500, "Error interno del servidor")

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "API DASS-21 funcionando correctamente",
        "version": "2.0",
        "predictor_loaded": predictor is not None,
        "openai_configured": bool(OPENAI_API_KEY),
        "timestamp": "2025-11-16T05:00:00Z"
    }

@app.get("/")
async def read_root():
    return {
        "message": "API DASS-21 Predictor", 
        "version": "2.0",
        "status": "active",
        "endpoints": {
            "/health": "GET - Estado del servicio",
            "/predict": "POST - Evaluación DASS-21", 
            "/chat": "POST - Chat de apoyo emocional"
        }
    }

# -----------------------------
# CONFIGURACIÓN RAILWAY
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")