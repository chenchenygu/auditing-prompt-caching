from enum import StrEnum

from clients.client import Client


class APIProviderName(StrEnum):
    """Enum for the names of API providers."""

    ANTHROPIC = "anthropic"
    AZURE = "azure"
    BEDROCK = "bedrock"
    COHERE = "cohere"
    DEEPINFRA = "deepinfra"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    GEMINI = "gemini"
    GROQ = "groq"
    HYPERBOLIC = "hyperbolic"
    LEPTON = "lepton"
    MISTRAL = "mistral"
    OCTOAI = "octoai"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    REPLICATE = "replicate"
    TOGETHER = "together"


def create_client_for_provider(
    *,
    provider: APIProviderName,
    api_key: str | None,
) -> Client:
    """Creates a new instance of the Client subclass corresponding to the given provider.

    Args:
        provider (APIProviderName): The name of the API provider.
        api_key (str | None): The API key to use. If None, the API key is read from
            the `client.API_KEY_NAME` environment variable.

    Returns:
        Client: A new instance of the Client subclass.
    """
    if provider == APIProviderName.ANTHROPIC:
        from clients.anthropic_client import AnthropicClient

        return AnthropicClient(api_key=api_key)

    elif provider == APIProviderName.AZURE:
        from clients.azure_client import AzureClient

        return AzureClient(api_key=api_key)

    elif provider == APIProviderName.BEDROCK:
        from clients.anthropic_bedrock_client import AnthropicBedrockClient

        return AnthropicBedrockClient(api_key=api_key)

    elif provider == APIProviderName.COHERE:
        from clients.cohere_client import CohereClient

        return CohereClient(api_key=api_key)

    elif provider == APIProviderName.DEEPINFRA:
        from clients.deepinfra_client import DeepInfraClient

        return DeepInfraClient(api_key=api_key)

    elif provider == APIProviderName.DEEPSEEK:
        from clients.deepseek_client import DeepSeekClient

        return DeepSeekClient(api_key=api_key)

    elif provider == APIProviderName.FIREWORKS:
        from clients.fireworks_client import FireworksClient

        return FireworksClient(api_key=api_key)

    elif provider == APIProviderName.GEMINI:
        from clients.gemini_client import GeminiClient

        return GeminiClient(api_key=api_key)

    elif provider == APIProviderName.GROQ:
        from clients.groq_client import GroqClient

        return GroqClient(api_key=api_key)

    elif provider == APIProviderName.HYPERBOLIC:
        from clients.hyperbolic_client import HyperbolicClient

        return HyperbolicClient(api_key=api_key)

    elif provider == APIProviderName.LEPTON:
        from clients.lepton_client import LeptonClient

        return LeptonClient(api_key=api_key)

    elif provider == APIProviderName.MISTRAL:
        from clients.mistral_client import MistralClient

        return MistralClient(api_key=api_key)

    elif provider == APIProviderName.OCTOAI:
        from clients.octoai_client import OctoAIClient

        return OctoAIClient(api_key=api_key)

    elif provider == APIProviderName.OPENAI:
        from clients.openai_client import OpenAIClient

        return OpenAIClient(api_key=api_key)

    elif provider == APIProviderName.PERPLEXITY:
        from clients.perplexity_client import PerplexityClient

        return PerplexityClient(api_key=api_key)

    elif provider == APIProviderName.REPLICATE:
        from clients.replicate_client import ReplicateClient

        return ReplicateClient(api_key=api_key)

    elif provider == APIProviderName.TOGETHER:
        from clients.together_client import TogetherClient

        return TogetherClient(api_key=api_key)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_client_class_for_provider(provider: APIProviderName) -> type[Client]:
    """Returns the Client subclass corresponding to the given provider.

    Args:
        provider (APIProviderName): The name of the API provider.

    Returns:
        type[Client]: The Client subclass.
    """
    if provider == APIProviderName.ANTHROPIC:
        from clients.anthropic_client import AnthropicClient

        return AnthropicClient

    elif provider == APIProviderName.AZURE:
        from clients.azure_client import AzureClient

        return AzureClient

    elif provider == APIProviderName.BEDROCK:
        from clients.anthropic_bedrock_client import AnthropicBedrockClient

        return AnthropicBedrockClient

    elif provider == APIProviderName.COHERE:
        from clients.cohere_client import CohereClient

        return CohereClient

    elif provider == APIProviderName.DEEPINFRA:
        from clients.deepinfra_client import DeepInfraClient

        return DeepInfraClient

    elif provider == APIProviderName.DEEPSEEK:
        from clients.deepseek_client import DeepSeekClient

        return DeepSeekClient

    elif provider == APIProviderName.FIREWORKS:
        from clients.fireworks_client import FireworksClient

        return FireworksClient

    elif provider == APIProviderName.GEMINI:
        from clients.gemini_client import GeminiClient

        return GeminiClient

    elif provider == APIProviderName.GROQ:
        from clients.groq_client import GroqClient

        return GroqClient

    elif provider == APIProviderName.HYPERBOLIC:
        from clients.hyperbolic_client import HyperbolicClient

        return HyperbolicClient

    elif provider == APIProviderName.LEPTON:
        from clients.lepton_client import LeptonClient

        return LeptonClient

    elif provider == APIProviderName.MISTRAL:
        from clients.mistral_client import MistralClient

        return MistralClient

    elif provider == APIProviderName.OCTOAI:
        from clients.octoai_client import OctoAIClient

        return OctoAIClient

    elif provider == APIProviderName.OPENAI:
        from clients.openai_client import OpenAIClient

        return OpenAIClient

    elif provider == APIProviderName.PERPLEXITY:
        from clients.perplexity_client import PerplexityClient

        return PerplexityClient

    elif provider == APIProviderName.REPLICATE:
        from clients.replicate_client import ReplicateClient

        return ReplicateClient

    elif provider == APIProviderName.TOGETHER:
        from clients.together_client import TogetherClient

        return TogetherClient

    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_api_key_name_for_provider(provider: APIProviderName) -> str:
    """Returns the environment variable name for the API key for the given provider.

    Args:
        provider (APIProviderName): The name of the API provider.

    Returns:
        str: The name of the environment variable.
    """
    # noinspection PyTypeChecker
    return get_client_class_for_provider(provider).API_KEY_NAME
