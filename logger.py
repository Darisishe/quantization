import logging
import os

# Custom stream to redirect stdout to logger
class LoggerStream:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message.strip():
            self.logger.info(message)

    def flush(self):
        pass

# Function to set up logging
def setup_logging(log_name):
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Define log file path
    log_file = f"{log_dir}/{log_name}.log"

    # Create and configure logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)

    # Define log format
    formatter = logging.Formatter("[%(asctime)s]: %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger
