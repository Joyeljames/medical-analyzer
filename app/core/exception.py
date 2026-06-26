from app.core.logger import get_logger

logger = get_logger(__name__)

class MedicalAnalyzerException(Exception):
    def __init__(self,message:str):
        self.message = message
        logger.error(f"MedicalAnalyzerException:{message}")
        super().__init__(self.message)


class PDFloadError(MedicalAnalyzerException):
    """Raised When PDF cannot be loaded or read"""
    pass

class PDFEmptyError(MedicalAnalyzerException):
    """Raised when PDF has no Extractable text"""
    pass

class ChunkingError(MedicalAnalyzerException):
    """Raised when text spliting fails"""
    pass

class EmbeddingError(MedicalAnalyzerException):
    """Raised When embeeding generation fails"""
    pass

class RetrieverError(MedicalAnalyzerException):
    """Raised when faiss retrival fails"""
    pass

class ChatbotError(MedicalAnalyzerException):
    """Raised when llama 3 fails to response"""
    pass
