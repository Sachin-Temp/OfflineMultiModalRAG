import time
from llama_cpp import Llama
from pathlib import Path

model_path = "models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
if not Path(model_path).exists():
    print(f"Model {model_path} NOT FOUND")
    exit(1)

print(f"Loading model: {model_path}")
start_load = time.time()
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=512,
    verbose=True
)
print(f"Model loaded in {time.time() - start_load:.2f}s")

prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nSay 'Hello' and tell me a short joke.\n<|assistant|>\n"
print("Starting inference...")
start_infer = time.time()
output = llm(prompt, max_tokens=50)
print(f"Inference finished in {time.time() - start_infer:.2f}s")
print(f"Output: {output['choices'][0]['text']}")
