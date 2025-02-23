import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import EmbeddingsAPITimingData


class HyperbolicClient(OpenAICompatibleClient):
    """Client for the Hyperbolic API.

    The Hyperbolic API is compatible with the OpenAI API.

    https://docs.hyperbolic.xyz/docs/rest-api
    """

    API_KEY_NAME = "HYPERBOLIC_API_KEY"
    BASE_URL = "https://api.hyperbolic.xyz/v1"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        """Hyperbolic does not return server processing time in the response headers, so returns None."""
        return None

    def time_embeddings_api_request(
        self,
        *,
        prompt: str,
        model: str,
    ) -> EmbeddingsAPITimingData:
        raise NotImplementedError("Hyperbolic does not have an embeddings API.")
