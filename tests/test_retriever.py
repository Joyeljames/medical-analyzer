import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pdf_loader import load_pdf
from app.services.chunker import chunk_documents
from app.services.embedder import create_and_save_embeddings
from app.services.retriever import get_retriever, retreive_relevent_chunk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(BASE_DIR, "data", "sample_medical_report.pdf")


# Step 1 — Full pipeline so far
docs = load_pdf(pdf_path)
chunks = chunk_documents(docs)
vectorstore = create_and_save_embeddings(chunks=chunks)
print(f"pipeline ready {len(chunks)}chunks is embedded....")


# Step 2 — Create retriever
retriever = get_retriever(vectorestore=vectorstore,k=3)
print("✅ Retriever created")

# Step 3 — Test multiple queries
test_queries = [
    "What is my cholesterol level?",
    "Do I have anemia?",
    "What did the doctor recommend?"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)

    results  =retreive_relevent_chunk(retriever=retriever,query=query)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1} (Page {doc.metadata.get('page')}):")
        print(f"  {doc.page_content[:120]}...")

