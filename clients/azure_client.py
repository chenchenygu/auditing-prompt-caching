import logging

import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import ChatAPITimingData, EmbeddingsAPITimingData

logger = logging.getLogger(__name__)


class AzureClient(OpenAICompatibleClient):
    """Client for the Azure OpenAI API.

    The Azure OpenAI API is compatible with the OpenAI API.

    https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
    """

    API_KEY_NAME = "AZURE_API_KEY"
    BASE_URL = ""

    def __init__(
        self,
        api_key: str | None,
    ) -> None:
        """Initializes the AzureClient.

        Args:
            api_key (str | None): The API key to use. If None,
                it will be set to the AZURE_API_KEY environment variable.
                Should be in the format "API_KEY ENDPOINT".
        """
        super().__init__(api_key)
        parts = self.api_key.split()
        if len(parts) != 2:
            raise ValueError("API key must contain two parts separated by a space")
        self.api_key, self.endpoint = parts
        logger.info(f"Azure endpoint: {self.endpoint}")

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        """Azure does not return server processing time in the response headers, so returns None."""
        return None

    def _get_headers(self) -> dict[str, str]:
        if "openai" in self.endpoint:
            return {
                "api-key": self.api_key,
            }
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def time_chat_api_request(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int = 1,
        temperature: float = 1.0,
    ) -> ChatAPITimingData:
        headers = self._get_headers()
        return self._time_chat_api_request_url(
            prompt=prompt,
            model=model,
            url=self.endpoint,
            max_tokens=max_tokens,
            temperature=temperature,
            headers=headers,
        )

    def time_embeddings_api_request(
        self,
        *,
        prompt: str,
        model: str,
    ) -> EmbeddingsAPITimingData:
        headers = self._get_headers()
        return self._time_embeddings_api_request_url(
            prompt=prompt,
            model=model,
            url=self.endpoint,
            headers=headers,
        )
