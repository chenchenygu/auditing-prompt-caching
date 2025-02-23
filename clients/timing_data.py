import dataclasses
import json
import pprint
from dataclasses import dataclass
from typing import Self

import httpx
import requests

_ABBREVIATED_STR_PREFIX_LENGTH = 32
_ABBREVIATED_STR_SUFFIX_LENGTH = 32
_ABBREVIATED_LIST_PREFIX_LENGTH = 16
_MAX_STR_DISPLAY_LENGTH = 150
_MAX_LIST_DISPLAY_LENGTH = 50
MILLISECONDS_PER_SECOND = 1000


def abbreviate_str(s: str, prefix_len: int, suffix_len: int) -> str:
    """Abbreviates a string by replacing the middle part with ellipses.

    Args:
        s (str): The string to abbreviate.
        prefix_len (int): The number of characters to keep at the start of the string.
        suffix_len (int): The number of characters to keep at the end of the string.

    Returns:
        str: The abbreviated string.

    >>> abbreviate_str("hello world", 3, 3)
    'hel...rld'
    """
    if len(s) <= prefix_len + suffix_len:
        return s
    return f"{s[:prefix_len]}...{s[len(s) - suffix_len :]}"


def abbreviate_dict_values(d: dict) -> dict:
    """Abbreviates the values of a dictionary, recursing into nested dictionaries.

    Args:
        d (dict): The dictionary to abbreviate.

    Returns:
        dict: The abbreviated dictionary.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = abbreviate_dict_values(v)
        elif isinstance(v, str) and len(v) > _MAX_STR_DISPLAY_LENGTH:
            d[k] = abbreviate_str(
                v,
                _ABBREVIATED_STR_PREFIX_LENGTH,
                _ABBREVIATED_STR_SUFFIX_LENGTH,
            )
        elif isinstance(v, list):
            # we only want to abbreviate embeddings, which are lists of floats
            if len(v) > _MAX_LIST_DISPLAY_LENGTH and isinstance(v[0], float):
                d[k] = v[:_ABBREVIATED_LIST_PREFIX_LENGTH]
            for i, elem in enumerate(d[k]):
                if isinstance(elem, dict):
                    d[k][i] = abbreviate_dict_values(elem)
    return d


@dataclass(frozen=True, kw_only=True)
class RequestData:
    """Dataclass for HTTP request data."""

    url: str
    method: str
    headers: dict[str, str]
    body: dict

    @classmethod
    def from_request(
        cls,
        r: requests.PreparedRequest | httpx.Request,
        *,
        api_key: str | None,
        abbreviated_api_key: str = "",
    ) -> Self:
        """Creates a RequestData object from a request object.

        Args:
            r (requests.PreparedRequest): The request object.
            api_key (str | None): The API key to abbreviate, to avoid storing it in the data.
                If None, no abbreviation is performed.
            abbreviated_api_key (str): The abbreviated API key. Defaults to the empty string.

        Returns:
            RequestData: The request data.
        """
        headers = dict(r.headers)
        if api_key is not None:
            for k, v in headers.items():
                if isinstance(v, str) and api_key in v:
                    headers[k] = v.replace(api_key, abbreviated_api_key)
        if isinstance(r, requests.PreparedRequest):
            body_bytes = r.body
        elif isinstance(r, httpx.Request):
            body_bytes = r.content
        else:
            raise ValueError(f"Unsupported request type: {type(r)}")
        return cls(
            url=str(r.url),
            method=r.method,
            headers=headers,
            body=json.loads(body_bytes.decode()),
        )


@dataclass(frozen=True, kw_only=True)
class APITimingData:
    """
    Dataclass for timing data for API requests.

    Attributes:
        client_time (float): The client-side response time for the API request, in seconds.
        server_time (float | None): The server-side processing time for the API request,
            in seconds. None if the server time is not available.
        prompt (str): The prompt sent to the API.
        n_prompt_tokens (int): The number of tokens in the prompt.
        sent_timestamp (float): The Unix timestamp when the request was sent.
        model (str): The model used for the API request.
        api_key (str): The (abbreviated) API key used for the request.
        request (RequestData | dict): The HTTP request data.
        response_headers (dict): The HTTP headers in the API response.
        response (dict): The raw API response.
    """

    client_time: float
    server_time: float | None
    prompt: str
    n_prompt_tokens: int
    sent_timestamp: float
    model: str
    api_key: str
    request: RequestData | dict
    response_headers: dict
    response: dict

    def __str__(self) -> str:
        d = dataclasses.asdict(self)
        d = abbreviate_dict_values(d)
        return f"{self.__class__.__name__}\n{pprint.pformat(d, width=100, compact=True, sort_dicts=False)}"


@dataclass(frozen=True, kw_only=True)
class ChatAPITimingData(APITimingData):
    """Dataclass for timing data for chat API requests.

    Attributes:
        completion (str): The completion returned by the API.
        n_completion_tokens (int): The number of tokens in the completion.
    """

    completion: str
    n_completion_tokens: int


@dataclass(frozen=True, kw_only=True)
class EmbeddingsAPITimingData(APITimingData):
    """Dataclass for timing data for embeddings API requests.

    Attributes:
        embedding (list[float]): The embedding returned by the API.
    """

    embedding: list[float]
