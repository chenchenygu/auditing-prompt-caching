import requests

from clients.openai_compatible_client import OpenAICompatibleClient


class DeepInfraClient(OpenAICompatibleClient):
    """Client for the DeepInfra API.

    The DeepInfra API is compatible with the OpenAI API.

    https://deepinfra.com/docs/openai_api
    """

    API_KEY_NAME = "DEEPINFRA_API_KEY"
    BASE_URL = "https://api.deepinfra.com/v1/openai"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        """DeepInfra does not return server processing time in the response headers, so returns None."""
        return None
