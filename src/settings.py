import os
import dotenv

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)
VIDEO_PATH = os.environ.get("VIDEO_PATH")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH")
ROBOFLOW_KEY = os.environ.get("ROBOFLOW_KEY")
