import os

from pathlib import Path

from dotenv import load_dotenv

from google import genai
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)