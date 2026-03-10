import time
from rag.rag_assistant import ask_with_context

print("\nLocal AI Assistant")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        break

    start = time.time()

    print("\nAI:\n")

    sources = ask_with_context(question)

    end = time.time()

    print("\nSources:", sources)

    print(f"\nResponse time: {end - start:.2f} seconds\n")