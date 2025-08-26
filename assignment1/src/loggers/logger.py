import logging
import os

LOG_DIR_FOLDER = "data/logs"
if not os.path.exists(LOG_DIR_FOLDER):
    os.makedirs(LOG_DIR_FOLDER)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # Write logs to a file
        logging.FileHandler(f"{LOG_DIR_FOLDER}/app.log"),
        logging.StreamHandler(),  # Print logs to console
    ],
)

logger = logging.getLogger(__name__)
