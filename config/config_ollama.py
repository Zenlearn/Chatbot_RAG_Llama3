from langchain_ollama import OllamaLLM
from config.config_env import LLAMA_MODEL

llama = OllamaLLM(model=LLAMA_MODEL)