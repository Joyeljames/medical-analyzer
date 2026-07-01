from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.exception import ChatbotError
from app.core.logger import get_logger

logger = get_logger(__name__)
MODEL_NAME = "llama3"

MEDICAL_PROMPT_TEMPLATE = """
You are a helpful medical report assistant.
Your job is to help patients understand their medical reports clearly.

Use ONLY the information provided in the context below to answer the question.
If the answer is not found in the context, say "I could not find that information in your medical report."
Never guess, assume, or use outside knowledge.
Keep your answer clear, simple and easy for a patient to understand.

Context from Medical Report:
{context}

Patient Question: {question}

Answer:
"""
def get_llm()->OllamaLLM:
    """
    Loads and returns the Ollama Llama3 LLM
    """
    try:
        logger.info(f"Loading LLM model: {MODEL_NAME}")

        llm = OllamaLLM(model=MODEL_NAME,
                        temperature=0.1)
        logger.info(f"✅ LLM model {MODEL_NAME} loaded successfully")
        return llm
    
    except Exception as e:
        logger.error(f"Failed to load LLM model {MODEL_NAME}: {e}")
        raise ChatbotError(f"Failed to load LLM model {MODEL_NAME}: {e}")
    
def format_docs(docs:list)->str:
        """
        Joins retrieved chunks into single context string
        """

        return "\n\n".join([doc.page_content for doc in docs])
    
    

def build_rag_chain(retriever:BaseRetriever):

        """
        Builds a RAG chain using the provided retriever and the LLM
        """

        try:
            llm = get_llm()
            prompt = PromptTemplate(
                template = MEDICAL_PROMPT_TEMPLATE,
                input_variables = ["context", "question"]
            )

            rag_chain =(
                {
                    "context":retriever | format_docs,
                    "question":RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            logger.info("✅ RAG chain built successfully")
            return rag_chain
        
        except ChatbotError:
            raise

        except Exception as e:
            logger.exception(f"Failed to build RAG chain: {e}")
            raise ChatbotError(f"Failed to build RAG chain: {str(e)}")

        

def ask_questions(ragchain,question:str)->str:
    """
    Asks a question to the RAG chain and returns the answer
    """
    logger.info(f"Asking question: {question}")
    try:
        if not question or not question.strip():
            raise ChatbotError("Question cannot be empty or whitespace.")
        
        answer = ragchain.invoke(question)
        logger.info(f"Received answer: {answer}")
        return answer
    except ChatbotError:
        raise
    except Exception as e:
        logger.exception(f"Failed to ask question: {e}")
        raise ChatbotError(f"Failed to ask question: {str(e)}")


        



    
