import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.pdf_loader import load_pdf
from app.services.chunker import chunk_documents
from app.services.embedder import create_and_save_embeddings
from app.services.retriever import get_retriever
from app.services.chatbot import build_rag_chain, ask_questions


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(BASE_DIR, "data", "sample_medical_report.pdf")

print("setting up pipeline....")
docs = load_pdf(pdf_path)
chunks = chunk_documents(docs)
vectorstore = create_and_save_embeddings(chunks)
retriever = get_retriever(vectorstore)
ragchain = build_rag_chain(retriever)
print("pipeline setup complete. Asking questions....")

#test questions
questions = [
    "What is my cholesterol level?",
    "Do I have anemia?",
    "What did the doctor recommend?",
    "What is my blood sugar level?",  # Not in report — tests honest response
    "does i have cancer?"

]

for question in questions:
    print(f"{'='*60}")
    print(f"❓ Question: {question}")
    print(f"{'='*60}")
    answer = ask_questions(ragchain, question)
    print(f"💬 Answer: {answer}")
    print()
