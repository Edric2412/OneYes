import os
import sys

# Add the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Setup logging as early as possible
import logger_config
logger_config.setup_logging()

from main import main

# Run the Streamlit application
if __name__ == "__main__":
    main()
