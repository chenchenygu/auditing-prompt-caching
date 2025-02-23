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
from clients.timing_data import ChatAPITimingData, RequestData

logger = logging.getLogger(__name__)


class AnthropicClient(Client):
    """Client for the Anthropic API.

    https://docs.anthropic.com/en/api/
    """

    API_KEY_NAME = "ANTHROPIC_API_KEY"
    BASE_URL = "https://api.anthropic.com/v1"
    CHAT_ENDPOINT = "/messages"
    ANTHROPIC_VERSION = "2023-06-01"

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
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
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

        # sometimes anthropic returns an empty content list, but status 200
        try:
            completion = response_json["content"][0]["text"]
        except IndexError:
            completion = ""
            logger.warning(f"Empty content list in response: {response_json}")

        return ChatAPITimingData(
            client_time=client_time,
            server_time=None,
            prompt=prompt,
            n_prompt_tokens=response_json["usage"]["input_tokens"],
            completion=completion,
            n_completion_tokens=response_json["usage"]["output_tokens"],
            sent_timestamp=sent_timestamp,
            model=response_json["model"],
            api_key=self._get_abbreviated_api_key(),
            request=request_data,
            response_headers=dict(response.headers),
            response=response_json,
        )
