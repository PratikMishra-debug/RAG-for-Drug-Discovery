import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_ollama import OllamaLLM


# Load dataset
data = pd.read_csv("../data/drug_data.csv")


# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create Vector Database
vector_db = Chroma.from_texts(
    texts=data["answer"].tolist(),
    embedding=embedding
)


# User Question
query = input("Ask about drug: ")


# Retrieve context
docs = vector_db.similarity_search(
    query,
    k=2
)


context = "\n".join(
    [doc.page_content for doc in docs]
)


# Llama model
llm = OllamaLLM(
    model="llama3.2:1b"
)


prompt = f"""

Use only this context:

{context}

Question:
{query}

Answer:
"""


response = llm.invoke(prompt)


print("\n--- Retrieved Context ---")
print(context)

print("\n--- Final Answer ---")
print(response)

import ollama


context = result["documents"][0]


prompt = f"""
You are a medical assistant.

Use only the given context to answer.

Context:
{context}

Question:
{query}

Answer:
"""


response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role":"user",
            "content":prompt
        }
    ]
)


print("\nFinal Answer:")
print(response["message"]["content"])