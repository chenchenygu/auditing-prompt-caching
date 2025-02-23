import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import MILLISECONDS_PER_SECOND, ChatAPITimingData, EmbeddingsAPITimingData


class LeptonClient(OpenAICompatibleClient):
    """Client for the Lepton API.

    The Lepton API is compatible with the OpenAI API., except that each model
    has its own base URL.

    https://www.lepton.ai/references/llm_models
    """

    API_KEY_NAME = "LEPTON_API_KEY"
    BASE_URL = "https://{model}.lepton.run/api/v1"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        # https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/router_filter#x-envoy-upstream-service-time
        return float(response.headers["x-envoy-upstream-service-time"]) / MILLISECONDS_PER_SECOND

    def time_chat_api_request(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int = 1,
        temperature: float = 1.0,
    ) -> ChatAPITimingData:
        base_url = self.BASE_URL.format(model=model)
        url = f"{base_url}{self.CHAT_ENDPOINT}"
        return self._time_chat_api_request_url(
            prompt=prompt,
            model=model,
            url=url,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def time_embeddings_api_request(
        self,
        *,
        prompt: str,
        model: str,
    ) -> EmbeddingsAPITimingData:
        raise NotImplementedError("Lepton does not have an embeddings API.")
