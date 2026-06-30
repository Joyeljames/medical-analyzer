import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.pdf_loader import load_pdf
from app.services.chunker import chunk_documents
from app.services.embedder import create_and_save_embeddings, load_embeddings


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(BASE_DIR, "data", "sample_medical_report.pdf")


#load pdf
docs = load_pdf(pdf_path)
print(f"pdf loaded : {len(docs)}")

#chunks
chunks = chunk_documents(docs)
print(f"chunks created {len(chunks)}")

#create embedding
vectorestore = create_and_save_embeddings(chunks)
print("Faiss index created sucessfully")

query = "what is my cholestrol level?"
results =vectorestore.similarity_search(query,k=3)

print(f"\n--- Top 3 Similar Chunks for: '{query}' ---")

for i,result in enumerate(results):
    print(f"\nResults {i+1} :")
    print(f" Page : {result.metadata.get('page')}")
    print(f"Content : {result.page_content[:100]}....")

print("\n---- Testing load from disk ----")
loded_vs=load_embeddings()
results2 = loded_vs.similarity_search(query, k=2)
print(f"✅ Loaded from disk — {len(results2)} results found")
