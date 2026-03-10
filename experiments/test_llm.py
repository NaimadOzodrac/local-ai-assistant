from llm.client import ask_llm

print("Asking local model...\n")

question = "Explain tango philosophy in one paragraph"

answer = ask_llm(question)

print("\nAI:\n")
print(answer)