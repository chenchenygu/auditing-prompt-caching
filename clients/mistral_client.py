import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import MILLISECONDS_PER_SECOND


class MistralClient(OpenAICompatibleClient):
    """Client for the Mistral API.

    The Mistral API is compatible with the OpenAI API.

    https://docs.mistral.ai/api/
    """

    API_KEY_NAME = "MISTRAL_API_KEY"
    BASE_URL = "https://api.mistral.ai/v1"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        # https://docs.konghq.com/gateway/latest/how-kong-works/routing-traffic/#response
        return float(response.headers["x-kong-upstream-latency"]) / MILLISECONDS_PER_SECOND
