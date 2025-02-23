import argparse
import dataclasses
import json
import logging
import os
import pprint
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from clients.client import Client, EndpointType
from clients.client_factory import APIProviderName, create_client_for_provider, get_api_key_name_for_provider
from clients.timing_data import APITimingData
from utils.json_utils import DataclassAndNumpyJSONEncoder
from utils.prompt_utils import generate_attacker_prompt, generate_random_prompt
from utils.stats_utils import compute_stats

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

VICTIM_API_KEY_SUFFIX = "_VICTIM"
ATTACKER_PER_ORG_API_KEY_SUFFIX = "_ATTACKER_PER_ORG"
ATTACKER_GLOBAL_API_KEY_SUFFIX = "_ATTACKER_GLOBAL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharing_level", type=str, choices=[t for t in SharingLevel], required=True)
    parser.add_argument("--provider", type=str, choices=[p for p in APIProviderName], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--endpoint", type=str, choices=[e for e in EndpointType], required=True)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--n_prompt_tokens", type=int, required=True)
    parser.add_argument("--prefix_fraction", type=float, required=True)
    parser.add_argument("--n_victim_requests", type=int, required=True)
    parser.add_argument("--victim_max_tokens", type=int, default=1)
    parser.add_argument(
        "--sleep_time",
        type=float,
        default=1.0,
        help="Time to sleep between requests to avoid hitting rate limits.",
    )
    parser.add_argument("--max_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--console_logging_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--file_logging_level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help=(
            "File to write logs to. If not provided, logs are not written to a file. "
            "Set to 'auto' to automatically create a log file in the `logs/` directory "
            "with the same name and path as the output file, but with a `.log` extension."
        ),
    )
    parser.add_argument("--output_file", type=str, help="(Optional) Output file to write data to.")
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="data",
        help=(
            "(Optional) Base directory to write data to. Defaults to `data`. "
            "Subdirectories will be automatically created."
        ),
    )
    parser.add_argument(
        "--full_output_dir",
        type=str,
        help="(Optional) Full directory to write data to. No subdirectories will be created.",
    )
    parser.add_argument("--env_files", type=str, nargs="*")
    return parser.parse_args()


class SharingLevel(StrEnum):
    """Enum for the levels of cache sharing."""

    PER_USER = "per_user"
    PER_ORG = "per_org"
    GLOBAL = "global"


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """Shared configuration for the cache timing experiments."""

    provider: APIProviderName
    model: str
    endpoint: EndpointType
    sharing_level: SharingLevel
    n_samples: int
    n_prompt_tokens: int
    prefix_fraction: float
    n_victim_requests: int
    sleep_time: float
    victim_max_tokens: int
    max_tokens: int = 1
    temperature: float = 1.0


def send_victim_requests(
    *,
    config: ExperimentConfig,
    prompt: str,
    victim_client: Client,
) -> list[APITimingData]:
    """Send victim API requests."""
    logger.debug("Victim API requests")
    victim_responses = []
    for _ in range(config.n_victim_requests):
        try:
            timing_data = victim_client.time_api_request(
                prompt=prompt,
                model=config.model,
                endpoint=config.endpoint,
                max_tokens=config.victim_max_tokens,
                temperature=config.temperature,
            )
            logger.debug("Victim timing data: %s", timing_data)
            victim_responses.append(timing_data)
        except Exception:
            logger.exception("Victim API request failed")
        time.sleep(config.sleep_time)
    return victim_responses


def produce_cache_hit(
    *,
    config: ExperimentConfig,
    victim_client: Client,
    attacker_client: Client,
    cache_hit_data: list[APITimingData],
    victim_data: list[list[APITimingData]],
) -> None:
    """Run procedure to attempt to produce a cache hit."""
    victim_prompt = generate_random_prompt(config.n_prompt_tokens)
    logger.debug("Victim prompt: %s", victim_prompt)
    victim_data_inner_list = send_victim_requests(
        config=config,
        prompt=victim_prompt,
        victim_client=victim_client,
    )

    attacker_prompt = generate_attacker_prompt(
        victim_prompt,
        config.prefix_fraction,
    )
    try:
        timing_data = attacker_client.time_api_request(
            prompt=attacker_prompt,
            model=config.model,
            endpoint=config.endpoint,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        tqdm.write(f"Cache hit: client_time = {timing_data.client_time}, server_time = {timing_data.server_time}")
        logger.debug("Attacker cache hit timing data: %s", timing_data)
        cache_hit_data.append(timing_data)
        victim_data.append(victim_data_inner_list)
    except Exception:
        logger.exception("Attacker cache hit API request failed")


def produce_cache_miss(
    *,
    config: ExperimentConfig,
    attacker_client: Client,
    cache_miss_data: list[APITimingData],
) -> None:
    """Runs procedure to produce a cache miss."""
    prompt = generate_random_prompt(config.n_prompt_tokens)
    try:
        timing_data = attacker_client.time_api_request(
            prompt=prompt,
            model=config.model,
            endpoint=config.endpoint,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        tqdm.write(f"Cache miss: client_time = {timing_data.client_time}, server_time = {timing_data.server_time}")
        logger.debug("Attacker cache miss timing data: %s", timing_data)
        cache_miss_data.append(timing_data)
    except Exception:
        logger.exception("Attacker cache miss API request failed")


def collect_timing_data(
    *,
    config: ExperimentConfig,
    victim_client: Client,
    attacker_client: Client,
) -> dict:
    """Collects timing data for cache hits and cache misses."""
    cache_hit_data: list[APITimingData] = []
    cache_miss_data: list[APITimingData] = []
    victim_data: list[list[APITimingData]] = []

    sample_order = ["cache_hit"] * config.n_samples + ["cache_miss"] * config.n_samples
    random.shuffle(sample_order)
    for sample_type in tqdm(sample_order):
        logger.debug(sample_type)
        if sample_type == "cache_hit":
            produce_cache_hit(
                config=config,
                victim_client=victim_client,
                attacker_client=attacker_client,
                cache_hit_data=cache_hit_data,
                victim_data=victim_data,
            )
        else:
            produce_cache_miss(
                config=config,
                attacker_client=attacker_client,
                cache_miss_data=cache_miss_data,
            )
        time.sleep(config.sleep_time)

    data = {
        "cache_hit": cache_hit_data,
        "cache_miss": cache_miss_data,
        "victim": victim_data,
    }
    stats_dict = compute_stats(cache_hit_data=cache_hit_data, cache_miss_data=cache_miss_data)
    stats_str = f"Statistics: {pprint.pformat(stats_dict, width=100, sort_dicts=False)}"
    print(stats_str)
    logger.info(stats_str)
    data["stats"] = stats_dict
    return data


def create_clients(
    config: ExperimentConfig,
) -> tuple[Client, Client]:
    """Creates the victim and attacker clients.

    Args:
        config (ExperimentConfig): The experiment configuration.

    Returns:
        tuple[Client, Client]: The victim and attacker clients, in that order.
    """
    client_api_key_name = get_api_key_name_for_provider(config.provider)

    victim_api_key = os.environ[f"{client_api_key_name}{VICTIM_API_KEY_SUFFIX}"]
    victim_client = create_client_for_provider(provider=config.provider, api_key=victim_api_key)

    if config.sharing_level == SharingLevel.PER_USER:
        return victim_client, victim_client

    if config.sharing_level == SharingLevel.PER_ORG:
        attacker_api_key = os.environ[f"{client_api_key_name}{ATTACKER_PER_ORG_API_KEY_SUFFIX}"]
    elif config.sharing_level == SharingLevel.GLOBAL:
        attacker_api_key = os.environ[f"{client_api_key_name}{ATTACKER_GLOBAL_API_KEY_SUFFIX}"]
    else:
        raise ValueError(f"Unknown attacker type: {config.sharing_level}")

    if victim_api_key == attacker_api_key:
        raise Exception(f"Victim and attacker API keys must be different in the {config.sharing_level} sharing level.")
    attacker_client = create_client_for_provider(provider=config.provider, api_key=attacker_api_key)
    return victim_client, attacker_client


def run_audit(
    config: ExperimentConfig,
) -> dict:
    """Runs the cache timing experiment."""
    total_prompt_tokens = (config.n_victim_requests + 2) * config.n_prompt_tokens * config.n_samples
    logger.info(f"Total number of prompt tokens: {total_prompt_tokens:,}")
    max_total_completion_tokens = config.n_samples * (
        config.n_victim_requests * config.victim_max_tokens + 2 * config.max_tokens
    )
    logger.info(f"Max total number of completion tokens: {max_total_completion_tokens:,}")

    victim_client, attacker_client = create_clients(config)
    data = collect_timing_data(
        config=config,
        victim_client=victim_client,
        attacker_client=attacker_client,
    )
    return data


def get_output_path(
    config: ExperimentConfig,
    *,
    base_output_dir: str | os.PathLike[str] = "data",
) -> Path:
    """Returns the output path for the experiment data."""
    output_dir = Path(base_output_dir) / config.provider / f"{config.sharing_level}"
    model = config.model.split("/")[-1]
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    timestamp = timestamp.replace("+00:00", "Z").replace(":", "")
    prefix_fraction_str = f"{config.prefix_fraction:.3g}".replace(".", "-")
    return output_dir / (
        f"{config.provider}_{model}_{config.sharing_level}_"
        f"n{config.n_samples}_p{config.n_prompt_tokens}_"
        f"pf{prefix_fraction_str}_v{config.n_victim_requests}_"
        f"{timestamp}.json"
    )


def main():
    args = parse_args()

    formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.console_logging_level)
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    provider = APIProviderName(args.provider)
    endpoint = EndpointType(args.endpoint)
    sharing_level = SharingLevel(args.sharing_level)

    config = ExperimentConfig(
        provider=provider,
        model=args.model,
        endpoint=endpoint,
        sharing_level=sharing_level,
        n_samples=args.n_samples,
        n_prompt_tokens=args.n_prompt_tokens,
        prefix_fraction=args.prefix_fraction,
        n_victim_requests=args.n_victim_requests,
        max_tokens=args.max_tokens,
        victim_max_tokens=args.victim_max_tokens,
        temperature=args.temperature,
        sleep_time=args.sleep_time,
    )
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = get_output_path(config, base_output_dir=args.base_output_dir)

    if args.full_output_dir:
        output_file = Path(args.full_output_dir) / output_file.name

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        raise FileExistsError(f"Output file {output_file} already exists")

    if not args.log_file:
        log_file = None
        logger.info("Not writing to log file")
    elif args.log_file == "auto":
        log_file = Path("logs") / output_file.with_suffix(".log")
    else:
        log_file = Path(args.log_file)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch(exist_ok=False)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(args.file_logging_level)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to {log_file}")

    if not args.env_files:
        load_dotenv(verbose=True)  # load from .env file
        load_dotenv(f".env.{provider}", verbose=True, override=True)  # load from .env.<provider> file
    else:
        logger.info(f"Loading environment variables from {args.env_files}")
        for env_file in args.env_files:
            load_dotenv(env_file, verbose=True, override=True)

    logger.info(f"Config: {config}")

    data = run_audit(config)

    data["args"] = vars(args)
    data["config"] = dataclasses.asdict(config)

    logger.info(f"Writing data to {output_file}")
    output_file.touch(exist_ok=False)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, cls=DataclassAndNumpyJSONEncoder)

    print(f"Log file: {log_file}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
