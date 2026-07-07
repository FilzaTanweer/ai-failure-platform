import logging
import sys
from src.core.config import settings

def configure_logging() -> logging.Logger:
    """
    Configures and returns a highly readable, standardized logger for the platform.
    Adapts seamlessly between detailed debugging outputs and clean production streams.
    """
    logger = logging.getLogger(settings.PROJECT_NAME)
    
    # Prevent duplicate handler injection on multiple calls
    if logger.hasHandlers():
        return logger
        
    # Configure level based on debug flags
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    logger.setLevel(log_level)
    
    # Professional stream formatting for terminal readability
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Setup standard output console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized successfully in {settings.APP_ENV} mode.")
    return logger

# Globally accessible logger instance
app_logger = configure_logging()