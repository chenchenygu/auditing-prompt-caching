import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import EmbeddingsAPITimingData


class DeepSeekClient(OpenAICompatibleClient):
    """Client for the DeepSeek API.

    The DeepSeek API is compatible with the OpenAI API, but it also returns
    the number of cache hit and cache miss tokens.

    https://platform.deepseek.com/api-docs/api/create-chat-completion/
    """

    API_KEY_NAME = "DEEPSEEK_API_KEY"
    BASE_URL = "https://api.deepseek.com"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        """DeepSeek does not return server processing time in the response headers, so returns None."""
        return None

    def time_embeddings_api_request(
        self,
        *,
        prompt: str,
        model: str,
    ) -> EmbeddingsAPITimingData:
        raise NotImplementedError("DeepSeek does not have an embeddings API.")
