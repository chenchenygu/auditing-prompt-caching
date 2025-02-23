import logging
import time

import httpx
from anthropic import AnthropicBedrock
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from clients.client import Client
from clients.timing_data import MILLISECONDS_PER_SECOND, ChatAPITimingData, RequestData

logger = logging.getLogger(__name__)


class AnthropicBedrockClient(Client):
    """Client for the Anthropic Amazon Bedrock API.

    https://docs.anthropic.com/en/api/claude-on-amazon-bedrock
    """

    API_KEY_NAME = "BEDROCK_API_KEY"

    def __init__(
        self,
        api_key: str | None,
    ) -> None:
        """Initializes the AnthropicBedrockClient.

        The AWS region is obtained from the AWS_REGION environment variable.

        Args:
            api_key (str | None): The API key to use. If None,
                it will be set to the BEDROCK_API_KEY environment variable.
                Should be in the format "AWS_ACCESS_KEY AWS_SECRET_KEY".
        """
        super().__init__(api_key)
        parts = self.api_key.split()
        if len(parts) != 2:
            raise ValueError("API key must contain two parts separated by a space")
        self.aws_access_key, self.aws_secret_key = parts

    @staticmethod
    def _parse_server_time(response: httpx.Response) -> float | None:
        """Parses the server processing time (in seconds) from the API response.

        Args:
            response (httpx.Response): The response object from the API request.

        Returns:
            float | None: The server processing time in seconds, or None if it is not available.
        """
        try:
            return float(response.headers["x-amzn-bedrock-invocation-latency"]) / MILLISECONDS_PER_SECOND
        except KeyError:
            logger.warning("Could not parse server time from response headers %s", response.headers)
            return None

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
        # need to create the client each time to avoid multiprocessing/threading issues
        client = AnthropicBedrock(
            aws_access_key=self.aws_access_key,
            aws_secret_key=self.aws_secret_key,
            timeout=self._TIMEOUT,
            max_retries=0,
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        sent_timestamp = time.time()
        response = client.messages.with_raw_response.create(**payload)

        client_time = response.elapsed.total_seconds()
        server_time = self._parse_server_time(response)
        response_json = response.http_response.json()
        request_data = RequestData.from_request(
            response.http_request,
            api_key=self.api_key,
            abbreviated_api_key=self._get_abbreviated_api_key(),
        )
        try:
            completion = response_json["content"][0]["text"]
        except IndexError:
            completion = ""
            logger.warning(f"Empty content list in response: {response_json}")

        return ChatAPITimingData(
            client_time=client_time,
            server_time=server_time,
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
