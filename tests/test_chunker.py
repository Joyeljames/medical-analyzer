import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pdf_loader import load_pdf
from app.services.chunker import chunk_documents

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(BASE_DIR, "data", "sample_medical_report.pdf")

docs  =load_pdf(pdf_path)
print(f"\n pages loaded : {len(docs)}")

chunks = chunk_documents(docs)
print(f"total chunks created {len(chunks)}")

print("\n--- First 3 Chunks Preview ---")
for i,chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(f"  Page     : {chunk.metadata.get('page')}")
    print(f"  Length   : {len(chunk.page_content)} characters")
    print(f"  Content  : {chunk.page_content[:100]}...")

