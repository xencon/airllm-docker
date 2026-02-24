import json
import logging
import asyncio
import torch
from threading import Thread
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TextIteratorStreamer
)

# --- Logging & Init ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Optimized AirLLM Server")

# --- Global State ---
# We use a queue to handle requests one by one to avoid VRAM collisions
request_queue = asyncio.Queue()
model = None
tokenizer = None
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=attn_impl,
        trust_remote_code=True
    )
    model.eval()

# --- The Inference Worker ---
async def inference_worker():
    """Background task that processes the queue sequentially."""
    while True:
        request_data, response_queue = await request_queue.get()
        try:
            await process_inference(request_data, response_queue)
        except Exception as e:
            logger.error(f"Worker Error: {e}")
            await response_queue.put(None) # Signal error
        finally:
            request_queue.task_done()

async def process_inference(request, response_queue):
    user_prompt = request.messages[-1].content
    inputs = tokenizer(user_prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=request.max_tokens,
        do_sample=request.temperature > 0,
        temperature=request.temperature,
    )

    # Offload generation to a thread so we can await the streamer
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        await response_queue.put(new_text)
    
    await response_queue.put(None) # Signal completion

# --- API Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Create a private queue for this specific request's tokens
    token_queue = asyncio.Queue()
    await request_queue.put((request, token_queue))

    async def stream_generator():
        while True:
            token = await token_queue.get()
            if token is None: break
            
            chunk = {
                "id": "chatcmpl-local",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.on_event("startup")
async def startup():
    load_model()
    asyncio.create_task(inference_worker())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)
