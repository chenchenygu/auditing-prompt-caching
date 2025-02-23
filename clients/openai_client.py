import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import MILLISECONDS_PER_SECOND


class OpenAIClient(OpenAICompatibleClient):
    """Client for the OpenAI API.

    https://platform.openai.com/docs/api-reference
    """

    API_KEY_NAME = "OPENAI_API_KEY"
    BASE_URL = "https://api.openai.com/v1"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        return float(response.headers["openai-processing-ms"]) / MILLISECONDS_PER_SECOND
