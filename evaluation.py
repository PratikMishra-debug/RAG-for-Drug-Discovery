import pandas as pd


baseline = pd.read_csv("results/baseline_results.csv")
rag = pd.read_csv("results/rag_results.csv")


print("Total Questions:", len(baseline))


# Simple comparison
comparison = pd.DataFrame()

comparison["Question"] = baseline["question"]

comparison["Baseline Answer"] = baseline["answer"]

comparison["RAG Answer"] = rag["answer"]


comparison.to_csv(
    "results/comparison.csv",
    index=False
)


print("\nComparison file created ✅")
print(comparison.head())