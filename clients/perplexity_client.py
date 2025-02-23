import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import EmbeddingsAPITimingData


class PerplexityClient(OpenAICompatibleClient):
    """Client for the Perplexity API.

    The Perplexity API is compatible with the OpenAI API.

    https://docs.perplexity.ai/docs/getting-started
    https://docs.perplexity.ai/reference/post_chat_completions
    """

    API_KEY_NAME = "PERPLEXITY_API_KEY"
    BASE_URL = "https://api.perplexity.ai"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        """Perplexity does not return server processing time in the response headers, so returns None."""
        return None

    def time_embeddings_api_request(
        self,
        *,
        prompt: str,
        model: str,
    ) -> EmbeddingsAPITimingData:
        raise NotImplementedError("Perplexity does not have an embeddings API.")
