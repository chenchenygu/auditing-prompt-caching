import logging

import numpy as np
import scipy

from clients.timing_data import APITimingData

logger = logging.getLogger(__name__)


def get_client_time(data: APITimingData) -> float:
    return data.client_time


def get_server_time(data: APITimingData) -> float:
    return data.server_time


def get_time_fns_dict() -> dict[str, callable]:
    return {
        "client_time": get_client_time,
        "server_time": get_server_time,
    }


def compute_stats(
    *,
    cache_hit_data: list[APITimingData],
    cache_miss_data: list[APITimingData],
) -> dict:
    """Computes statistics for independent timing data."""
    stats_dict = {}
    for time_name, time_fn in get_time_fns_dict().items():
        try:
            stats_inner_dict = {}
            cache_hit_times = [time_fn(d) for d in cache_hit_data if time_fn(d) is not None]
            cache_miss_times = [time_fn(d) for d in cache_miss_data if time_fn(d) is not None]
            if not cache_hit_times or not cache_miss_times:
                logger.warning(f"No timing data for {time_name}")
                continue
            cache_hit_times = np.array(cache_hit_times)
            cache_miss_times = np.array(cache_miss_times)

            stats_inner_dict["median"] = {
                "cache_hit": np.median(cache_hit_times),
                "cache_miss": np.median(cache_miss_times),
            }
            stats_inner_dict["mean"] = {
                "cache_hit": np.mean(cache_hit_times),
                "cache_miss": np.mean(cache_miss_times),
            }
            stats_inner_dict["std"] = {
                "cache_hit": np.std(cache_hit_times),
                "cache_miss": np.std(cache_miss_times),
            }
            stats_inner_dict["cache_proportion_faster_than_min_no_cache"] = np.mean(
                cache_hit_times < np.min(cache_miss_times)
            )
            stats_inner_dict["ks_2samp"] = scipy.stats.ks_2samp(
                cache_hit_times,
                cache_miss_times,
                alternative="greater",
            )._asdict()
            stats_dict[time_name] = stats_inner_dict
        except Exception:
            logger.exception(f"Could not compute statistics using {time_name}")
    return stats_dict
