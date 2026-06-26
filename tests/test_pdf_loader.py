from app.services.pdf_loader import load_pdf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.pdf_loader import load_pdf

# Use absolute path — no confusion
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(BASE_DIR, "data", "sample_medical_report.pdf")

docs = load_pdf(pdf_path)

for doc in docs:
    print(f"Page {doc.metadata['page']}:{doc.page_content}")
