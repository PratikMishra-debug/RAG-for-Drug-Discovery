import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb


# Load dataset
data = pd.read_csv("data/drug_dataset.csv")


# Create text
documents = data["question"] + " " + data["answer"]


# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create embeddings
embeddings = model.encode(documents.tolist())


# Chroma database
client = chromadb.Client()

collection = client.create_collection(
    name="drug_knowledge"
)


# Store data
for i in range(len(documents)):
    collection.add(
        ids=[str(i)],
        embeddings=[embeddings[i].tolist()],
        documents=[documents[i]]
    )


print("Drug knowledge base created!")


# Test query
query = "What is aspirin used for?"

query_embedding = model.encode([query])


result = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=2
)


print("\nRetrieved:")
print(result["documents"])