import os
from google import genai

class GeminiEnricher:
    def __init__(self, api_key: str = None):
        # Uses passed API key or falls back to GEMINI_API_KEY environment variable
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=key)

    def enrich(self, plugin_id: str, readme_content: str) -> str:
        prompt = f"Analyze and enrich metadata for plugin '{plugin_id}' based on its README:\n\n{readme_content}"
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
