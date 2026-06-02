import logging
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = "logs"
log_dir_path = BASE_DIR / LOG_DIR
os.makedirs(log_dir_path, exist_ok=True)

LOG_FILE = log_dir_path / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logger = logging.getLogger("bosch_logger")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
)

# File handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)