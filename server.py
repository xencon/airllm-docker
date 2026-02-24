import json
import logging
import asyncio
import torch
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
app = FastAPI(title="AirLLM NVMe Optimized Server")

# --- Global State ---
request_queue = asyncio.Queue()
model = None
tokenizer = None
MODEL_PATH = "/app/models"

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

            # Start generation in a background thread
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # Iterate over the streamer and push to the local response queue
            for new_text in streamer:
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

@app.on_event("startup")
async def startup_event():
    load_model()
    # Launch the persistent worker task
    asyncio.create_task(inference_worker())

if __name__ == "__main__":
    import uvicorn
    # Use standard uvicorn runner
    uvicorn.run(app, host="0.0.0.0", port=11434)
