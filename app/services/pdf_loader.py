import fitz
from pathlib import Path
from langchain_core.documents import Document
from app.core.logger import get_logger
from app.core.exception import PDFloadError,PDFEmptyError

logger  = get_logger(__name__)

def load_pdf(file_path)-> list[Document]:

    """
    Reads a PDF and returns a list of LangChain Document objects.
    Each page becomes one Document with metadata.

    """
    logger.info(f"starting pdf load: {file_path}")
    path = Path(file_path)

    try:
        if not path.exists():
            raise PDFloadError(f"file does not exist: {file_path}")
        
        if path.suffix.lower()!=".pdf":
            raise PDFloadError(f"file is not a pdf: {file_path}")
        
        pdf = fitz.open(str(path))
        documents =[]

        for pagenum in range(len(pdf)):
            page = pdf[pagenum]
            text = page.get_text()

            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata ={
                        "source" : str(path),
                        "page" : pagenum + 1,
                        "Total_pages" : len(pdf) 
                    }
                )
                documents.append(doc)
                logger.info(f"page {pagenum +1 } loaded - {len(text)} characters")
            
            pdf.close()

            if not documents:
                raise PDFEmptyError(f"No extractable text found in : {file_path}")
            
            logger.info(f"pdf load completer - {len(documents)} page extracted")
            return documents
        
    except (PDFEmptyError,PDFloadError):
        raise

    except Exception as e:
                logger.exception(f"Unexpected error while loading PDF: {e}")
                raise PDFloadError(f"Unexpected error: {str(e)}") 