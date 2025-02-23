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


class ReplicateClient(Client):
    """Client for the Replicate API.

    https://replicate.com/meta/meta-llama-3-8b-instruct/api
    """

    API_KEY_NAME = "REPLICATE_API_KEY"
    BASE_URL = "https://api.replicate.com/v1"
    CHAT_ENDPOINT = "/models/{model}/predictions"

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
        url = f"{self.BASE_URL}{self.CHAT_ENDPOINT.format(model=model)}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }
        sent_timestamp = time.time()
        create_response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self._TIMEOUT,
        )
        create_response.raise_for_status()
        create_response_json = create_response.json()
        if create_response_json["error"] is not None:
            raise Exception(create_response_json["error"])
        request_data = RequestData.from_request(
            create_response.request,
            api_key=self.api_key,
            abbreviated_api_key=self._get_abbreviated_api_key(),
        )

        # completion needs to be retrieved from a different URL,
        # so we can't measure client-side timing
        get_url = create_response_json["urls"]["get"]
        while True:
            if time.time() - sent_timestamp > self._TIMEOUT:
                raise Exception("Timed out waiting for completion")
            # get prediction endpoint has rate limit of 3000 requests per minute
            # https://replicate.com/docs/reference/http#rate-limits
            time.sleep(0.1)
            try:
                get_response = requests.get(
                    get_url,
                    headers=headers,
                    timeout=self._TIMEOUT,
                )
                get_response.raise_for_status()
                response_json = get_response.json()
                if response_json["status"] == "succeeded":
                    break
            except Exception:
                logger.exception("Error getting prediction")
                continue

        # noinspection PyUnboundLocalVariable
        server_time = response_json["metrics"]["time_to_first_token"]
        # noinspection PyUnboundLocalVariable
        response_headers = dict(get_response.headers)

        return ChatAPITimingData(
            client_time=server_time,  # we can't measure client-side timing
            server_time=server_time,
            prompt=prompt,
            n_prompt_tokens=response_json["metrics"]["input_token_count"],
            completion="".join(response_json["output"]),
            n_completion_tokens=response_json["metrics"]["output_token_count"],
            sent_timestamp=sent_timestamp,
            model=model,
            api_key=self._get_abbreviated_api_key(),
            request=request_data,
            response_headers=response_headers,
            response=response_json,
        )
