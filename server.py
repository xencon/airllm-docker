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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

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
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = True

def load_model():
    global model, tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

async def inference_worker():
    while True:
        item = await request_queue.get()
        if item is None: continue
        request_data, response_queue = item
        try:
            user_prompt = request_data.messages[-1].content
            inputs = tokenizer(user_prompt, return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=request_data.max_tokens,
                do_sample=request_data.temperature > 0,
                temperature=request_data.temperature,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                await response_queue.put(new_text)
                # This is the "secret sauce" to force a flush
                await asyncio.sleep(0) 
            
            await response_queue.put(None)
        except Exception as e:
            logger.error(f"Worker Error: {e}")
            await response_queue.put(None)
        finally:
            request_queue.task_done()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    token_queue = asyncio.Queue()
    await request_queue.put((request, token_queue))

    async def stream_generator():
        while True:
            token = await token_queue.get()
            if token is None: break
            
            chunk = {
                "choices": [{"delta": {"content": token}, "index": 0}]
            }
            # Add the \n\n required by the SSE protocol to trigger a client-side flush
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.on_event("startup")
async def startup():
    load_model()
    asyncio.create_task(inference_worker())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)
