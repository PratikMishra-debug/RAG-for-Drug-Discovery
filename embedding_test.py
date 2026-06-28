import os

print(os.getcwd())

from sentence_transformers import SentenceTransformer
import pandas as pd

# Load dataset
data = pd.read_csv(
  pd.read_csv("data/drug_data.csv")
)

print("Dataset:")
print(data.head())


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Convert text into vectors
embeddings = model.encode(
    data["answer"].tolist()
)


print("\nEmbedding shape:")
print(embeddings.shape)