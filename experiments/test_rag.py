import time
from rag.rag_assistant import ask_with_context

question = "Cual es el nucleo de crianza al que le agradece el autor?"

print("\nQuestion:", question)

start = time.time()

answer, sources = ask_with_context(question)

end = time.time()

elapsed = end - start

print("\nAI:\n")
print(answer)

print("\nSources:")
print(sources)

print(f"\nResponse time: {elapsed:.2f} seconds")