# server.py
import json
import logging
import os
import torch
from threading import Thread
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TextIteratorStreamer
)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Optimized AirLLM Server")

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

# ----------------------------
# OpenAI-Compatible Models
# ----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

# ----------------------------
# Model Initialization
# ----------------------------
MODEL_PATH = "/app/models"

# --- Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.2 # Lower for coding
    stream: Optional[bool] = True

# --- Hardware Optimizations ---
def load_model():
    global model, tokenizer
    logger.info("Initializing high-performance AirLLM instance...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        # OPTIMIZATION: Quantize the KV Cache to 4-bit to support 
        # much longer context (crucial for large code files)
        bnb_4bit_kv_cache_quant=True 
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # OPTIMIZATION: Use FlashAttention-2 if hardware supports it
    # This speeds up the 'pre-fill' stage by up to 3x
    attn_impl = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa"

    # Note: AirLLM logic typically uses AutoModelForCausalLM with device_map="auto"
    # to handle the layer-sharding across disk/RAM/VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=attn_impl,
        trust_remote_code=True
    )
    model.eval()
    logger.info("Model loaded successfully!")

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# ----------------------------
# OpenAI Chat Completion Endpoint
# ----------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Extract prompt from the last user message
    user_prompt = request.messages[-1].content
    inputs = tokenizer(user_prompt, return_tensors="pt").to(model.device)

    if request.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=request.temperature,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def openai_stream_generator():
            for new_text in streamer:
                # Format each chunk as a Server-Sent Event (SSE) for OpenAI compatibility
                chunk = {
                    "id": "chatcmpl-local",
                    "object": "chat.completion.chunk",
                    "created": 1234567,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": new_text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")

    else:
        # Non-streaming logic
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt if the model returns it
        response_content = generated_text.replace(user_prompt, "").strip()

        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 1234567,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

# ----------------------------
# Health Endpoints
# ----------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "local-model", "object": "model", "owned_by": "organization-owner"}]
    }

@app.get("/health")
async def health():
    return {"status": "healthy" if model else "not loaded"}

if __name__ == "__main__":
    import uvicorn
    # Important: host 0.0.0.0 is required for Docker access
    uvicorn.run(app, host="0.0.0.0", port=11434)
