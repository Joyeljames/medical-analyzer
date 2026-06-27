from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.logger import get_logger
from app.core.exception import ChunkingError

logger = get_logger(__name__)


def chunk_documents(documents:list[Document]) ->list[Document]:
     """
    Takes list of LangChain Documents (pages)
    Returns list of smaller chunked Documents
    """
     logger.info(f"starting chunking : {len(documents)} pages received")

     try:
          if not documents:
               raise ChunkingError("No documents provide for chunking")
          
          splitter =RecursiveCharacterTextSplitter(
               chunk_size=500,
               chunk_overlap =50,
               separators=["\n\n", "\n", ".", " ", ""]

          )

          chunks = splitter.split_documents(documents)

          if not chunks:
               raise ChunkingError("chunking prodeced no output")
          
          logger.info(f"chunking completed - {len(chunks)} created from {len(documents)}")

          for i, chunk in enumerate(chunks):
               logger.info(f"chunk {i+1} - page {chunk.metadata.get('page')} - {len(chunk.page_content)} characters")
        

          return chunks
     except ChunkingError:
          raise
     
     except Exception as e:
          logger.exception(f"Unexpected error during chunking: {e}")
          raise ChunkingError(f"Unexpected error: {str(e)}")
          
