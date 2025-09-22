from huggingface_hub import create_repo
from dotenv import load_dotenv, find_dotenv
import os
from huggingface_hub import login
from huggingface_hub import HfApi

load_dotenv(find_dotenv(), override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./",
    repo_id="iwswordpress/marcus",
    repo_type="dataset",
)
