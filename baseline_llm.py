import ollama


question = "What is aspirin used for?"


response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role":"user",
            "content":question
        }
    ]
)


print("Question:")
print(question)

print("\nAnswer:")
print(response["message"]["content"])