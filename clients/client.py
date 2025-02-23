import logging
import os
from abc import ABC, abstractmethod
from enum import StrEnum

from clients.timing_data import ChatAPITimingData, EmbeddingsAPITimingData, abbreviate_str

logger = logging.getLogger(__name__)


class EndpointType(StrEnum):
    """Enum for API endpoints."""

    CHAT = "chat"
    EMBEDDINGS = "embeddings"


class Client(ABC):
    """Abstract base class for API clients."""

    _ABBREVIATED_API_KEY_PREFIX_LENGTH = 8
    _ABBREVIATED_API_KEY_SUFFIX_LENGTH = 8
    _TIMEOUT = 100

    def __init__(
        self,
        api_key: str | None,
    ) -> None:
        """Initializes the Client with an API key.

        Args:
            api_key (str | None): The API key to use. If None,
                it will be set to the `API_KEY_NAME` environment variable.
        """
        if api_key is None:
            api_key = os.environ[self.API_KEY_NAME]
        self.api_key = api_key
        logger.info("API key: %s", self._get_abbreviated_api_key())

    # noinspection PyPep8Naming
    @property
    @abstractmethod
    def API_KEY_NAME(self) -> str:
        """Base name of the environment variable containing the API key."""
        ...

    def _get_abbreviated_api_key(self) -> str:
        """Abbreviates the API key for display purposes.

        Returns:
            str: The abbreviated API key.
        """
        return abbreviate_str(
            self.api_key,
            self._ABBREVIATED_API_KEY_PREFIX_LENGTH,
            self._ABBREVIATED_API_KEY_SUFFIX_LENGTH,
        )

    def time_chat_api_request(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int = 1,
        temperature: float = 1.0,
    ) -> ChatAPITimingData:
        """Times a chat API request.

        Args:
            prompt (str): The prompt to send to the API.
            model (str): The model to use for the API request.
            max_tokens (int, optional): Maximum number of tokens in the output. Defaults to 1.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.

        Returns:
            ChatAPITimingData: The timing data for the chat API request.

        Raises:
            NotImplementedError: The method is not implemented in the subclass.
        """
        raise NotImplementedError(f"time_chat_api_request not implemented in {self.__class__.__name__}")

    def time_embeddings_api_request(
        self,
        *,
        prompt: str,
        model: str,
    ) -> EmbeddingsAPITimingData:
        """Times a chat API request.

        Args:
            prompt (str): The prompt to send to the API.
            model (str): The model to use for the API request.

        Returns:
            EmbeddingsAPITimingData: The timing data for the chat API request.

        Raises:
            NotImplementedError: The method is not implemented in the subclass.
        """
        raise NotImplementedError(f"time_embeddings_api_request not implemented in {self.__class__.__name__}")

    def time_api_request(
        self,
        prompt: str,
        *,
        model: str,
        endpoint: EndpointType,
        max_tokens: int = 1,
        temperature: float = 1.0,
    ) -> ChatAPITimingData | EmbeddingsAPITimingData:
        """Convenience wrapper method for timing API requests to a given endpoint.

        Args:
            prompt (str): The prompt to send to the API.
            model (str): The model to use for the API request.
            endpoint (EndpointType): The API endpoint to use.
            max_tokens (int, optional): Maximum number of tokens in the output. Defaults to 1.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.

        Returns:
            ChatAPITimingData | EmbeddingsAPITimingData: The timing data for the API request.
        """
        if endpoint is EndpointType.CHAT:
            return self.time_chat_api_request(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif endpoint is EndpointType.EMBEDDINGS:
            return self.time_embeddings_api_request(
                prompt=prompt,
                model=model,
            )
        else:
            raise ValueError(f"Invalid endpoint: {endpoint}")
