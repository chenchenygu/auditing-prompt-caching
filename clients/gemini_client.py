import logging
import re
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


class GeminiClient(Client):
    """Client for the Gemini API.

    https://ai.google.dev/api/generate-content
    https://ai.google.dev/api/embeddings
    """

    API_KEY_NAME = "GEMINI_API_KEY"
    BASE_URL = "https://generativelanguage.googleapis.com/v1/models/{model}:{endpoint}"
    CHAT_ENDPOINT = "generateContent"
    EMBEDDINGS_ENDPOINT = "embedContent"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        """Parses the server processing time (in seconds) from the API response.

        Args:
            response (requests.Response): The response object from the API request.

        Returns:
            float | None: The server processing time in seconds, or None if it is not available.
        """
        # Server-Timing header looks like "gfet4t7; dur=304"
        # https://cloud.google.com/spanner/docs/latency-points#google-front-end-latency
        server_timing_header = response.headers.get("Server-Timing")
        if server_timing_header is None:
            logger.error("No Server-Timing header found in response headers %s", response.headers)
            return None

        # look for number following "dur=", either an integer or a float
        matches = re.findall(r"dur=(\d+\.?\d*)", server_timing_header)
        if not matches:
            logger.error("No dur= matches found in header %s", server_timing_header)
            return None
        if len(matches) > 1:
            logger.error("Multiple dur= matches found in header %s, returning first match", server_timing_header)
        return float(matches[0]) / MILLISECONDS_PER_SECOND

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
        url = self.BASE_URL.format(model=model, endpoint=self.CHAT_ENDPOINT)
        query_params = {
            "key": self.api_key,
        }
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        sent_timestamp = time.time()
        response = requests.post(
            url,
            params=query_params,
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
            candidate = response_json["candidates"][0]
            completion = candidate["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            logger.exception("Could not parse completion")
            completion = ""
        usage_metadata = response_json["usageMetadata"]

        return ChatAPITimingData(
            client_time=client_time,
            server_time=self._parse_server_time(response),
            prompt=prompt,
            n_prompt_tokens=usage_metadata["promptTokenCount"],
            completion=completion,
            n_completion_tokens=usage_metadata["candidatesTokenCount"],
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
        url = self.BASE_URL.format(model=model, endpoint=self.EMBEDDINGS_ENDPOINT)
        query_params = {
            "key": self.api_key,
        }
        payload = {
            "content": {
                "parts": [{"text": prompt}],
            }
        }
        sent_timestamp = time.time()
        response = requests.post(
            url,
            params=query_params,
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

        return EmbeddingsAPITimingData(
            client_time=client_time,
            server_time=self._parse_server_time(response),
            prompt=prompt,
            n_prompt_tokens=-1,  # google does not provide token counts for embeddings models
            embedding=response_json["embedding"]["values"],
            sent_timestamp=sent_timestamp,
            model=model,
            api_key=self._get_abbreviated_api_key(),
            request=request_data,
            response_headers=dict(response.headers),
            response=response_json,
        )
