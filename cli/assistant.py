from llm.client import ask_llm

print("\nLocal AI Assistant")
print("-------------------")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("\nGoodbye\n")
        break

    print("\nThinking...\n")

    answer = ask_llm(question)

    print("AI:", answer)
    print()