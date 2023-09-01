from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import time

app = FastAPI()

#name = "/mnt/repo/text-generation-webui/models/meta-llama_Llama-2-13b-chat-hf"
#name = "/mnt/repo/text-generation-webui/models/meta-llama_Llama-2-7b-chat-hf"
name = "/mnt/repo/text-generation-webui/models/meta-llama_Llama-2-70b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, 
    rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

@app.post("/generate")
async def generate_text(prompt: str, token_limit: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start_time = time.perf_counter()
    output = model.generate(**inputs, streamer=streamer, use_cache=False, max_new_tokens=token_limit)
    duration = round(time.perf_counter() - start_time, 2)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"text": text, "duration": duration}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
