import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
import chromadb


# Load questions
questions = pd.read_csv("data/test_questions.csv")


# Load drug dataset
data = pd.read_csv("data/drug_dataset.csv")


documents = data["question"] + " " + data["answer"]


# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create vector database
client = chromadb.Client()

collection = client.create_collection(
    name="experiment_drugs"
)


embeddings = model.encode(documents.tolist())


for i in range(len(documents)):
    collection.add(
        ids=[str(i)],
        embeddings=[embeddings[i].tolist()],
        documents=[documents.iloc[i]]
    )


baseline_results = []
rag_results = []


for q in questions["question"]:

    print("\nQuestion:", q)


    # -------- BASELINE LLM --------

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role":"user",
                "content":q
            }
        ]
    )

    baseline_answer = response["message"]["content"]


    baseline_results.append(
        [q, baseline_answer]
    )


    # -------- RAG --------

    query_embedding = model.encode([q])


    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=2
    )


    context = result["documents"][0]


    prompt = f"""
Use only this context:

{context}

Question:
{q}

Answer:
"""


    rag_response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )


    rag_answer = rag_response["message"]["content"]


    rag_results.append(
        [q, rag_answer]
    )


# Save results

pd.DataFrame(
    baseline_results,
    columns=["question","answer"]
).to_csv(
    "results/baseline_results.csv",
    index=False
)


pd.DataFrame(
    rag_results,
    columns=["question","answer"]
).to_csv(
    "results/rag_results.csv",
    index=False
)


print("\nEXPERIMENT COMPLETED 🔥")