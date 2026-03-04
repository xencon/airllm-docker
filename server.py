import json
import os
import logging
import asyncio
import torch
from contextlib import asynccontextmanager
from pathlib import Path
import threading
from threading import Thread
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TextIteratorStreamer
)

# --- Logging & Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Launch the background model loader and inference worker on startup."""
    # Start the model loading in a background thread so the API can bind immediately
    threading.Thread(target=load_model, daemon=True).start()
    asyncio.create_task(inference_worker())
    yield

app = FastAPI(title="AirLLM NVMe Optimized Server", lifespan=lifespan)

# --- Global State ---
request_queue = asyncio.Queue()
model = None
tokenizer = None
is_model_loaded = False

def _resolve_model_path() -> str:
    """Load model path from config.json if present, else fall back to /app/models."""
    config_path = Path("/app/config.json")
    if config_path.exists():
        try:
            with config_path.open() as f:
                cfg = json.load(f)
            model_name = cfg.get("model_name", "")
            if model_name:
                logger.info(f"Config loaded: model_name={model_name}")
                # Return the HuggingFace model identifier so transformers downloads/caches it
                return model_name
        except Exception as e:
            logger.warning(f"Failed to parse config.json, falling back to /app/models: {e}")
    logger.info("No config.json found, using default MODEL_PATH=/app/models")
    return "/app/models"

MODEL_PATH = _resolve_model_path()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = True

# --- Model Loading ---
def load_model():
    global model, tokenizer
    logger.info("Loading model from NVMe with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Enable Flash Attention 2 if the hardware supports it
    device_cap = torch.cuda.get_device_capability()
    attn_impl = "flash_attention_2" if device_cap[0] >= 8 else "sdpa"
    logger.info(f"Using attention implementation: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=attn_impl,
        trust_remote_code=True
    )
    model.eval()
    
    global is_model_loaded
    is_model_loaded = True
    logger.info("Model load complete. API is now ready for inference.")

# --- Background Worker ---
async def inference_worker():
    """Sequential processing loop to handle NVMe layer swapping."""
    while True:
        item = await request_queue.get()
        if item is None:
            continue
            
        request_data, response_queue = item
        try:
            # Pass all messages to retain Continue CLI context (e.g. file contents)
            messages = [{"role": m.role, "content": m.content} for m in request_data.messages]
            
            # Prepare inputs
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=request_data.max_tokens,
                do_sample=request_data.temperature > 0,
                temperature=request_data.temperature,
            )

            def _generate_with_error_handling(**kwargs):
                try:
                    model.generate(**kwargs)
                except Exception as e:
                    import asyncio
                    # The asyncio queue is not directly thread-safe, so we use call_soon_threadsafe
                    # however response_queue here is an asyncio.Queue, but we can't await in sync thread.
                    # Since TextIteratorStreamer is thread-safe via its own queue, we can just put the exception there.
                    streamer.text_queue.put(e)

            # Start generation in a background thread
            thread = Thread(target=_generate_with_error_handling, kwargs=generation_kwargs)
            thread.start()

            # Iterate over the streamer and push to the local response queue
            for new_text in streamer:
                # TextIteratorStreamer will give us the exception object directly via loop if we injected it
                if isinstance(new_text, Exception):
                    raise new_text
                if new_text:
                    await response_queue.put(new_text)
                # Yield to the event loop to ensure tokens are sent immediately
                await asyncio.sleep(0.01)
            
            # None signals the end of this specific stream
            await response_queue.put(None)
            
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            await response_queue.put(e)
        finally:
            request_queue.task_done()

# --- API Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible endpoint for Continue CLI."""
    if not is_model_loaded:
        # Return a graceful streaming response indicating the loading status
        async def loading_stream():
            chunk = {
                "id": "airllm-gen",
                "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {"content": "⏳ Model is currently loading into memory. Please wait a few minutes and try again...\n"},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(loading_stream(), media_type="text/event-stream")

    token_queue = asyncio.Queue()
    
    # Add request to the global processing queue
    await request_queue.put((request, token_queue))

    async def stream_generator():
        while True:
            token = await token_queue.get()
            if token is None:
                break
            if isinstance(token, Exception):
                yield f"data: {json.dumps({'error': str(token)})}\n\n"
                break
            
            # Wrap token in OpenAI-style JSON for Markdown rendering
            chunk = {
                "id": "airllm-gen",
                "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {"content": token},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            # SSE protocol requires 'data: ' prefix and double newlines
            yield f"data: {json.dumps(chunk)}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models list stub."""
    model_id = MODEL_PATH if MODEL_PATH != "/app/models" else "local-model"
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "owned_by": "airllm",
        }]
    }

if __name__ == "__main__":
    import uvicorn
    # Use standard uvicorn runner
    uvicorn.run(app, host="0.0.0.0", port=11434)
