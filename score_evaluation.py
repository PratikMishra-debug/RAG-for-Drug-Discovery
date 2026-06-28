# import pandas as pd


# data = pd.read_csv("results/comparison.csv")


# print("Total Questions:", len(data))


# # Check fake drug questions manually later
# print("\nQuestions:")
# print(data["Question"])

# baseline_hallucination = 0
# rag_hallucination = 0


# for x in data["Baseline Answer"]:
#     if "xyz" in x.lower() or "fake" in x.lower():
#         baseline_hallucination += 1


# for x in data["RAG Answer"]:
#     if "no information" in x.lower():
#         rag_hallucination += 1


# print("Baseline hallucination:", baseline_hallucination)
# print("RAG hallucination:", rag_hallucination)

import pandas as pd

data = pd.read_csv("results/comparison.csv")


baseline_hallucination = 0
rag_hallucination = 0


baseline_labels = []
rag_labels = []


for x in data["Baseline Answer"]:
    if "xyz" in x.lower() or "fake" in x.lower():
        baseline_labels.append("Yes")
        baseline_hallucination += 1
    else:
        baseline_labels.append("No")


for x in data["RAG Answer"]:
    if "no information" in x.lower():
        rag_labels.append("Yes")
        rag_hallucination += 1
    else:
        rag_labels.append("No")


data["Baseline Hallucination"] = baseline_labels
data["RAG Hallucination"] = rag_labels


data.to_csv(
    "results/final_results.csv",
    index=False
)


print("Baseline hallucination:", baseline_hallucination)
print("RAG hallucination:", rag_hallucination)

print("Final results saved ✅")