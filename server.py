# server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
import logging
import os

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

# ----------------------------
# Request/Response Models
# ----------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

# ----------------------------
# Model Initialization
# ----------------------------
MODEL_PATH = "/app/models"

try:
    logger.info(f"Loading model from {MODEL_PATH} in 4-bit mode...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load tokenizer from local folder
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load model with device_map="auto"
    # Note: model.to(device) is REMOVED because it conflicts with bitsandbytes/accelerate
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    model.eval()
    logger.info("Model loaded successfully!")

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# ----------------------------
# Root & Health Endpoints
# ----------------------------
@app.get("/")
async def root():
    return {
        "message": "Coding LLM Server is running",
        "model": MODEL_PATH,
        "status": "loaded" if model else "not loaded"
    }

@app.get("/health")
async def health():
    if model is None:
        return {"status": "degraded", "model": "not loaded"}
    return {"status": "healthy", "model": MODEL_PATH}

# ----------------------------
# Streaming Generation Endpoint
# ----------------------------
@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Move inputs to the same device as the model's first layer
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generation kwargs
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=request.max_length,
            do_sample=True,
            temperature=request.temperature,
            top_p=0.95,
        )

        # Run generation in a separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def token_generator():
            for new_text in streamer:
                yield new_text

        return StreamingResponse(token_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Streaming generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Non-streaming Generation Endpoint
# ----------------------------
@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                do_sample=True,
                temperature=request.temperature,
                top_p=0.95,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text, "model": MODEL_PATH}

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Model Info Endpoint
# ----------------------------
@app.get("/model/info")
async def model_info():
    return {
        "model": MODEL_PATH,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(model.device) if model else "N/A"
    }
