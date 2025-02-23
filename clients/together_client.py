import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import MILLISECONDS_PER_SECOND


class TogetherClient(OpenAICompatibleClient):
    """Client for the Together API.

    The Together API is compatible with the OpenAI API.

    https://docs.together.ai/reference/
    """

    API_KEY_NAME = "TOGETHER_API_KEY"
    BASE_URL = "https://api.together.xyz/v1"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        return float(response.headers["x-inference-time"]) / MILLISECONDS_PER_SECOND
