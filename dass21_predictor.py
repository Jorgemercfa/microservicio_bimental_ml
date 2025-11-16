import os
import json
import re
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict

# ==================== CONFIGURACIÓN ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dass21")

# Índices DASS-21
DEP_INDICES = [3, 5, 10, 13, 16, 17, 21]
ANS_INDICES = [2, 4, 7, 9, 15, 19, 20]
EST_INDICES = [1, 6, 8, 11, 12, 14, 18]

# Mapeo directo
LABEL_MAP_SPANISH = {
    "no": 0, "nunca": 0, "casi nunca": 0,
    "raras veces": 1, "ocasionalmente": 1, "algunas veces": 1, "sí, poco": 1, "si, poco": 1,
    "bastante": 2, "frecuentemente": 2, "usualmente": 2, "casi siempre": 2, "sí, bastante": 2, "si, bastante": 2,
    "siempre": 3, "sí, mucho": 3, "si, mucho": 3
}

# ==================== MODELOS ====================
class TextListRequest(BaseModel):
    respuestas: List[str]

class ChatRequest(BaseModel):
    message: str

# ==================== PREDICTOR ====================
class DASS21Predictor:
    def __init__(self):
        logger.info("✅ Predictor inicializado")
        
    def predict_question(self, text: str) -> int:
        text = str(text).lower().strip()
        return LABEL_MAP_SPANISH.get(text, 1)
    
    def predict_scores(self, respuestas: List[str]) -> Dict[str, int]:
        if len(respuestas) != 21:
            raise ValueError("Se requieren 21 respuestas")
        
        valores = [self.predict_question(r) for r in respuestas]
        
        return {
            "depresion": sum(valores[i-1] for i in DEP_INDICES),
            "ansiedad": sum(valores[i-1] for i in ANS_INDICES),
            "estres": sum(valores[i-1] for i in EST_INDICES),
            "respuestas": valores,
            "nota": "Sistema confiable"
        }

# ==================== APP FASTAPI ====================
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = DASS21Predictor()
    logger.info("🚀 API INICIADA")
    yield

app = FastAPI(title="DASS-21 API", version="3.0", lifespan=lifespan)

# CORS EXTENDIDO
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    return {"message": "API DASS-21", "status": "active"}

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "predictor": predictor is not None,
        "openai": bool(OPENAI_API_KEY)
    }

@app.post("/predict")
async def predict(request: TextListRequest):
    try:
        if len(request.respuestas) != 21:
            raise HTTPException(422, "21 respuestas requeridas")
        
        result = predictor.predict_scores(request.respuestas)
        return result
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

def get_openai_response(mensaje: str) -> str:
    if not OPENAI_API_KEY:
        return "Chat no configurado"
    
    try:
        response = requests.post(
            OPENAI_API_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "system", "content": "Eres un asistente empático de salud mental."}, {"role": "user", "content": mensaje}],
                "max_tokens": 200,
                "temperature": 0.7,
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return "Error temporal"
    except:
        return "Servicio no disponible"

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(400, "Mensaje vacío")
        
        respuesta = get_openai_response(request.message)
        return {"respuesta": respuesta}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, "Error interno")

# ==================== INICIO ====================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")