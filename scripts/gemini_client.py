import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class EnrichmentParseError(Exception):
    """Raised when structured response generation or parsing fails."""


class EnrichmentResult(BaseModel):
    description: str = Field(description="A clean, concise description of the plugin's purpose.")
    category: str = Field(description="Primary category classification for the plugin.")
    tags: list[str] = Field(description="List of relevant tags for the plugin.")


class GeminiEnricher:
    def __init__(self, api_key: str = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=key)

    def enrich(self, plugin_id: str, readme_content: str) -> EnrichmentResult:
        prompt = (
            f"Analyze and enrich metadata for plugin '{plugin_id}' based on its README:\n\n"
            f"{readme_content}"
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EnrichmentResult,
                    temperature=0.2,
                ),
            )

            if response.parsed:
                return response.parsed

            return EnrichmentResult.model_validate_json(response.text)

        except Exception as err:
            raise EnrichmentParseError(f"Failed to enrich plugin '{plugin_id}' with Gemini") from err
