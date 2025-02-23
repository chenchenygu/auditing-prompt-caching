import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import MILLISECONDS_PER_SECOND


class OctoAIClient(OpenAICompatibleClient):
    """Client for the OctoAI API.

    The OctoAI API is compatible with the OpenAI API.

    https://octo.ai/docs/api-reference/text-gen/create-chat-completion-stream
    https://octo.ai/docs/text-gen-solution/migration-from-openai
    """

    API_KEY_NAME = "OCTOAI_API_KEY"
    BASE_URL = "https://text.octoai.run/v1"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        # https://docs.konghq.com/gateway/latest/how-kong-works/routing-traffic/#response
        return float(response.headers["X-Kong-Upstream-Latency"]) / MILLISECONDS_PER_SECOND
