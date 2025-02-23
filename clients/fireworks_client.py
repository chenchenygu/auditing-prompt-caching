import logging
import re

import requests

from clients.openai_compatible_client import OpenAICompatibleClient
from clients.timing_data import MILLISECONDS_PER_SECOND

logger = logging.getLogger(__name__)


class FireworksClient(OpenAICompatibleClient):
    """Client for the Fireworks API.

    The Fireworks API is compatible with the OpenAI API.

    https://docs.fireworks.ai/api-reference/introduction
    """

    API_KEY_NAME = "FIREWORKS_API_KEY"
    BASE_URL = "https://api.fireworks.ai/inference/v1"

    @staticmethod
    def _parse_server_time(response: requests.Response) -> float | None:
        try:
            return float(response.headers["fireworks-server-processing-time"])
        except KeyError:
            # fireworks-server-processing-time header is not present in embeddings response
            logger.debug("No fireworks-server-processing-time header in response headers %s", response.headers)

            # Server-Timing header looks like total;dur=195.0;desc="Total Response Time"
            server_timing_header = response.headers["Server-Timing"]

            # look for number following "dur=", either an integer or a float
            matches = re.findall(r"dur=(\d+\.?\d*)", server_timing_header)
            if len(matches) > 1:
                logger.warning("Multiple dur= matches found in header %s, returning first match", server_timing_header)
            return float(matches[0]) / MILLISECONDS_PER_SECOND
