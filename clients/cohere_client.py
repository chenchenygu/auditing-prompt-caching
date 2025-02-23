import logging
import time

import requests
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from clients.client import Client
from clients.timing_data import MILLISECONDS_PER_SECOND, ChatAPITimingData, EmbeddingsAPITimingData, RequestData

logger = logging.getLogger(__name__)


class CohereClient(Client):
    """Client for the Cohere API.

    https://docs.cohere.com/reference/about
    """

    API_KEY_NAME = "COHERE_API_KEY"
    BASE_URL = "https://api.cohere.com/v1"
    CHAT_ENDPOINT = "/chat"
    EMBEDDINGS_ENDPOINT = "/embed"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        """Parses the server processing time (in seconds) from the API response.

        Args:
            response (requests.Response): The response object from the API request.

        Returns:
            float | None: The server processing time in seconds, or None if it is not available.
        """
        # https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/router_filter#x-envoy-upstream-service-time
        return float(response.headers["x-envoy-upstream-service-time"]) / MILLISECONDS_PER_SECOND

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def time_chat_api_request(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int = 1,
        temperature: float = 1.0,
    ) -> ChatAPITimingData:
        url = f"{self.BASE_URL}{self.CHAT_ENDPOINT}"
        headers = {
            "Authorization": f"bearer {self.api_key}",
        }
        payload = {
            "message": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        sent_timestamp = time.time()
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self._TIMEOUT,
        )
        response.raise_for_status()

        client_time = response.elapsed.total_seconds()
        response_json = response.json()
        request_data = RequestData.from_request(
            response.request,
            api_key=self.api_key,
            abbreviated_api_key=self._get_abbreviated_api_key(),
        )
        try:
            server_time = self._parse_server_time(response)
        except Exception:
            logger.exception("Could not parse server timing info")
            server_time = None

        return ChatAPITimingData(
            client_time=client_time,
            server_time=server_time,
            prompt=prompt,
            n_prompt_tokens=response_json["meta"]["tokens"]["input_tokens"],
            completion=response_json["text"],
            n_completion_tokens=response_json["meta"]["tokens"]["output_tokens"],
            sent_timestamp=sent_timestamp,
            model=model,
            api_key=self._get_abbreviated_api_key(),
            request=request_data,
            response_headers=dict(response.headers),
            response=response_json,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def time_embeddings_api_request(
        self,
        *,
        prompt: str,
        model: str,
    ) -> EmbeddingsAPITimingData:
        url = f"{self.BASE_URL}{self.EMBEDDINGS_ENDPOINT}"
        headers = {
            "Authorization": f"bearer {self.api_key}",
        }
        payload = {
            "texts": [prompt],
            "model": model,
            "input_type": "classification",  # arbitrarily chosen
        }
        sent_timestamp = time.time()
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self._TIMEOUT,
        )
        response.raise_for_status()

        client_time = response.elapsed.total_seconds()
        response_json = response.json()
        request_data = RequestData.from_request(
            response.request,
            api_key=self.api_key,
            abbreviated_api_key=self._get_abbreviated_api_key(),
        )
        try:
            server_time = self._parse_server_time(response)
        except Exception:
            logger.exception("Could not parse server timing info")
            server_time = None

        return EmbeddingsAPITimingData(
            client_time=client_time,
            server_time=server_time,
            prompt=prompt,
            n_prompt_tokens=response_json["meta"]["billed_units"]["input_tokens"],
            embedding=response_json["embeddings"][0],
            sent_timestamp=sent_timestamp,
            api_key=self._get_abbreviated_api_key(),
            request=request_data,
            model=model,
            response_headers=dict(response.headers),
            response=response_json,
        )
