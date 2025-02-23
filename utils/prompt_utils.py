import random
import string


def generate_random_prompt(n_tokens: int) -> str:
    """Generates a random alphabetic prompt.

    Generates a random prompt consisting of `n_tokens` alphabetic letters
    (lower and uppercase) separated by spaces. Since tokenizers usually
    split on whitespace, this should result in exactly one token per letter.

    Args:
        n_tokens: The number of tokens (letters) in the prompt.

    Returns:
        str: The generated prompt.

    >>> random.seed(0)
    >>> generate_random_prompt(10)
    'R N v n A v O p y E'
    """
    return " ".join(random.choices(string.ascii_letters, k=n_tokens))


def generate_attacker_prompt(
    prompt: str,
    prefix_fraction: float,
) -> str:
    """Generates an attacker prompt that shares a prefix with the original prompt.

    The length of the shared prefix is exactly `prefix_fraction` times the length of the original prompt.

    Args:
        prompt (str): The prompt to modify.
        prefix_fraction (float): The fraction of the original prompt to keep.

    Returns:
        str: The attacker prompt.

    >>> random.seed(0)
    >>> generate_attacker_prompt("a b c d e f", 1)
    'a b c d e f'
    >>> random.seed(0)
    >>> generate_attacker_prompt("a b c d e f", 0.5)
    'a b c R N v'
    >>> random.seed(0)
    >>> generate_attacker_prompt("a b c d e", 0.8)
    'a b c d R'
    >>> random.seed(0)
    >>> generate_attacker_prompt("a b c d e", 0)
    'R N v n A'
    """
    tokens = prompt.split()
    prefix_length = round(len(tokens) * prefix_fraction)
    prefix = tokens[:prefix_length]
    suffix_length = len(tokens) - prefix_length
    suffix = random.choices(string.ascii_letters, k=suffix_length)
    if suffix:
        # ensure that the first token of the suffix is different from the original token
        # this ensures that the length of the shared prefix is exactly `prefix_length`
        orig_token = tokens[-suffix_length]
        if suffix[0] == orig_token:
            suffix[0] = random.choice([s for s in string.ascii_letters if s != orig_token])
    return " ".join(prefix + suffix)
