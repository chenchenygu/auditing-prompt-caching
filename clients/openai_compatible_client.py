import logging
import time
from abc import abstractmethod

import requests
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from clients.client import Client
from clients.timing_data import ChatAPITimingData, EmbeddingsAPITimingData, RequestData

logger = logging.getLogger(__name__)


class OpenAICompatibleClient(Client):
    """Abstract base class for clients that are compatible with the OpenAI API.

    https://platform.openai.com/docs/api-reference
    """

    CHAT_ENDPOINT = "/chat/completions"
    EMBEDDINGS_ENDPOINT = "/embeddings"

    # noinspection PyPep8Naming
    @property
    @abstractmethod
    def BASE_URL(self) -> str:
        """The base URL for the API."""
        ...

    @staticmethod
    @abstractmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        """Parses the server processing time (in seconds) from the API response.

        Args:
            response (requests.Response): The response object from the API request.

        Returns:
            float | None: The server processing time in seconds, or None if it is not available.
        """
        ...

    def time_chat_api_request(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int = 1,
        temperature: float = 1.0,
    ) -> ChatAPITimingData:
        url = f"{self.BASE_URL}{self.CHAT_ENDPOINT}"
        return self._time_chat_api_request_url(
            prompt=prompt,
            model=model,
            url=url,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _time_chat_api_request_url(
        self,
        *,
        prompt: str,
        model: str,
        url: str,
        max_tokens: int = 1,
        temperature: float = 1.0,
        headers: dict[str, str] | None = None,
    ) -> ChatAPITimingData:
        if headers is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
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
            n_prompt_tokens=response_json["usage"]["prompt_tokens"],
            completion=response_json["choices"][0]["message"]["content"],
            n_completion_tokens=response_json["usage"]["completion_tokens"],
            sent_timestamp=sent_timestamp,
            model=response_json["model"],
            api_key=self._get_abbreviated_api_key(),
            request=request_data,
            response_headers=dict(response.headers),
            response=response_json,
        )

    def time_embeddings_api_request(
        self,
        *,
        prompt: str,
        model: str,
    ) -> EmbeddingsAPITimingData:
        url = f"{self.BASE_URL}{self.EMBEDDINGS_ENDPOINT}"
        return self._time_embeddings_api_request_url(
            model=model,
            prompt=prompt,
            url=url,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _time_embeddings_api_request_url(
        self,
        *,
        prompt: str,
        model: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> EmbeddingsAPITimingData:
        if headers is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
        payload = {
            "input": prompt,
            "model": model,
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
            n_prompt_tokens = response_json["usage"]["prompt_tokens"]
        except KeyError:
            n_prompt_tokens = -1

        try:
            server_time = self._parse_server_time(response)
        except Exception:
            logger.exception("Could not parse server timing info")
            server_time = None

        return EmbeddingsAPITimingData(
            client_time=client_time,
            server_time=server_time,
            prompt=prompt,
            n_prompt_tokens=n_prompt_tokens,
            embedding=response_json["data"][0]["embedding"],
            sent_timestamp=sent_timestamp,
            model=response_json["model"],
            api_key=self._get_abbreviated_api_key(),
            request=request_data,
            response_headers=dict(response.headers),
            response=response_json,
        )
