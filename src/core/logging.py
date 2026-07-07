import logging
import sys
from src.core.config import settings

def configure_logging():
    """Configures centralized log tracing based on active configuration states."""
    # 🔧 FIX: Swapped legacy settings.PROJECT_NAME for our modern production settings.APP_NAME
    logger = logging.getLogger(settings.APP_NAME)
    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

app_logger = configure_logging()