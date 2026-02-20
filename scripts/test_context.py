import time
from llama_cpp import Llama
from pathlib import Path

model_path = "models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
print(f"Loading model: {model_path}")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=True
)

# Mock context: 2000 tokens (simulating RAG)
context = "This is a context sentence. " * 200
prompt = f"<|system|>\nUse context below.\n<|context|>\n{context}\n<|user|>\nSummarize the context.\n<|assistant|>\n"

print(f"Starting inference with ~2000 tokens context...")
start_infer = time.time()
output = llm(prompt, max_tokens=100)
print(f"Inference finished in {time.time() - start_infer:.2f}s")
