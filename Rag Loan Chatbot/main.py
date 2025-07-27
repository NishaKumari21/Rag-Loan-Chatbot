from retriever import Retriever
from generator import Generator

retriever = Retriever()
documents = retriever.prepare_documents()
retriever.create_index()

generator = Generator(use_openai=True)

while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == 'exit':
        break
    context = retriever.retrieve_context(query)
    response = generator.generate_response(context, query)
    print("\nAnswer:", response)