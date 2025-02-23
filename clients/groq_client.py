import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import EmbeddingsAPITimingData


class GroqClient(OpenAICompatibleClient):
    """Client for the Groq API.

    The Groq API is compatible with the OpenAI API, but it also returns
    server-side timing data.

    https://console.groq.com/docs/api-reference
    """

    API_KEY_NAME = "GROQ_API_KEY"
    BASE_URL = "https://api.groq.com/openai/v1"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        return response.json()["usage"]["prompt_time"]

    def time_embeddings_api_request(
        self,
        *,
        prompt: str,
        model: str,
    ) -> EmbeddingsAPITimingData:
        raise NotImplementedError("Groq does not have an embeddings API.")
