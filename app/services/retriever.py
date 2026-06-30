from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from app.core.exception import RetrieverError
from app.core.logger import get_logger

logger = get_logger(__name__)

def get_retriever(vectorestore:FAISS, k:int = 3)->BaseRetriever:
    """
    Wraps a FAISS vectorstore into a LangChain Retriever
    using MMR search for diverse, relevant results
    """
    logger.info(f"creating retriever with k={k},search_kwargs=mmr")

    try:
        if vectorestore is None:
            raise RetrieverError("vectorstore is None - cannot create retriever")
        
        retriver = vectorestore.as_retriever(
            search_type ="mmr",
            search_kwargs = {
                "k":k,
                "fetch_k":k*4,
                "lambda_mult":0.7
            }
        )

        logger.info("Retreiver created sucessfully")
        return retriver
    
    except RetrieverError:
        raise

    except Exception as e:
        logger.exception(f"unexpected errror creating retriever: {e}")
        raise RetrieverError(f"unexpected error {str(e)}")
    

def retreive_relevent_chunk(retriever:BaseRetriever,query:str)->list:
    """
    Takes a user query, returns relevant chunks using the retriever
    """
    logger.info(f"Retrieving chunks for query: '{query}'")

    try:
        if not query or not query.strip():
            raise RetrieverError("querry cannot be empty")
        
        results = retriever.invoke(query)

        logger.info(f"Retreived {len(results)} relevent chunks")

        for i,doc in enumerate(results):
            page = doc.metadata.get('page','unknown')
            preview = doc.page_content[:60].replace("\n"," ")
            logger.info(f"chunk {i+1} - page {page} -- '{preview}....'")

        return results
    
    except RetrieverError:
        raise

    
    except Exception as e:
        logger.exception(f"Unexpected error during retrieval: {e}")
        raise RetrieverError(f"Unexpected error: {str(e)}")