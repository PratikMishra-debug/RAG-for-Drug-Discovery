from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3.2:1b"
)

response = llm.invoke(
    "What is aspirin used for?"
)

print(response)