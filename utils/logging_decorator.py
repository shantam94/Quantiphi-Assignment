# logging_decorator.py

import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
    handlers=[
        logging.FileHandler("app.log")
    ])

def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Exiting {func.__name__}")
        return result
    return wrapper