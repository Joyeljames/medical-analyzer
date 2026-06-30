import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import torch

from app.core.logger import get_logger
from app.core.exception import EmbeddingError

logger = get_logger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "embeddings", "faiss_index")

def get_embedding()->HuggingFaceBgeEmbeddings:
    """
    Loads and returns the sentence transformer embedding model
    """

    try:
        logger.info(f"loading embedding model {EMBEDDING_MODEL}")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model =HuggingFaceBgeEmbeddings(
            model_name = EMBEDDING_MODEL,
            model_kwargs = {"device":device},
            encode_kwargs = {"normalize_embeddings":True}
            )
        
        logger.info("embedding model loaded sucessfully")
        return model
    
    except Exception as e:
        logger.exception(f"failed to load embeeding model: {e}")
        raise EmbeddingError(f"failed to load embedding model {str(e)}")
    
def create_and_save_embeddings(chunks:list[Document]) ->FAISS:
    """
    Takes list of chunks, creates embeddings, saves FAISS index to disk
    """

    logger.info(f"creating embedding for {len(chunks)} chunks")

    try:
        if not chunks:
            raise EmbeddingError("no chunks provided for embeddings")
        
        os.makedirs("embeddings",exist_ok=True)

        model =get_embedding()

        logger.info(f"building faiss index from chunks")
        vectostore = FAISS.from_documents(
            documents=chunks,
            embedding=model
        )
        vectostore.save_local(FAISS_INDEX_PATH)
        logger.info(f"vectores saved in {FAISS_INDEX_PATH} sucessfully")

        return vectostore
    except EmbeddingError:
        raise

    except Exception as e:
        logger.exception(f"Unexpected error during embedding: {e}")
        raise EmbeddingError(f"Unexpected error: {str(e)}")
    
def load_embeddings()->FAISS:
    """
    Loads existing FAISS index from disk
    """
    logger.info(f"loading existing index {FAISS_INDEX_PATH}")

    try:
        if not os.path.exists(FAISS_INDEX_PATH):
            raise EmbeddingError(f"faiss index not fount at {FAISS_INDEX_PATH}")
        
        model = get_embedding()
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings=model,
            allow_dangerous_deserialization=True
        )

        logger.info("faiss index loads sucessfully")
        return vectorstore
    except EmbeddingError:
        raise
    except Exception as e:
        logger.exception(f"unexpected error while loading faiss index {e}")
        raise EmbeddingError(f"Unexpected error {str(e)}")


