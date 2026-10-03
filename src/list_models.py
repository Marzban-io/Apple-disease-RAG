"""List the Gemini models your API key can use for text generation."""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads GEMINI_API_KEY from the .env file
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(model.name.replace("models/", ""))