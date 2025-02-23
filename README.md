# Auditing Prompt Caching in Language Model APIs

This repository contains code and data for the paper [Auditing Prompt Caching in Language Model APIs](https://arxiv.org/abs/2502.07776) by [Chenchen Gu](https://chenchenygu.github.io/), [Xiang Lisa Li](https://xiangli1999.github.io/), [Rohith Kuditipudi](https://web.stanford.edu/~rohithk/), [Percy Liang](https://cs.stanford.edu/~pliang/), [Tatsunori Hashimoto](https://thashim.github.io/).

This code is intended solely for research purposes, and should not be used for any malicious or harmful purposes.

## Table of Contents

- [Installation](#installation)
- [Running Audits](#running-audits)
  - [Usage](#usage)
  - [API Keys](#api-keys)
- [Data](#data)
- [Citation](#citation)

## Installation

1. Create a conda environment:
```shell
conda create -n auditing-prompt-caching python=3.12
conda activate auditing-prompt-caching
```

2. Install packages:
```shell
pip install -r requirements.txt
```

3. (Optional) Install pre-commit hooks ([ruff](https://github.com/astral-sh/ruff) linting and formatting):
```shell
pre-commit install
```

## Running Audits

### Usage

Example usage to run audits:
```shell
python audit.py \
    --sharing_level {per_user,per_org,global} \
    --provider <provider> \
    --model <model> \
    --endpoint {chat,embeddings} \
    --n_samples 250 \
    --n_prompt_tokens 5000 \
    --prefix_fraction 0.95 \
    --n_victim_requests 1 \
    --sleep_time 1.0 \
    --base_output_dir data \
```

To view a help message showing all arguments:
```shell
python audit.py -h
```

Provider names are listed in [`clients/client_factory.py`](clients/client_factory.py).

### API Keys

API keys are loaded as environment variables from the `.env` and `.env.<provider>` (e.g., `.env.openai`) files. Alternatively, env files can be specified using the `--env_files` argument, in which case the previous two files will not be automatically loaded.

The base `API_KEY_NAME` for each provider can be found in each `<provider>_client.py` file in the [`clients`](clients) directory. These names follow the format `<PROVIDER>_API_KEY`, e.g., `OPENAI_API_KEY`. Then, the environment variable name for the victim's API key is obtained by adding a suffix of `_VICTIM` to the base API key name. The environment variable names for the attacker's API key in the per-organization and global sharing levels are obtained by adding suffixes of `_ATTACKER_PER_ORG` and `_ATTACKER_GLOBAL`, respectively. (In the per-user level, the attacker and victim are the same user, so they have the same API key).

For example, to set the API keys for OpenAI, `.env` or `.env.openai` would look like this:
```dotenv
OPENAI_API_KEY_VICTIM="sk-..."
OPENAI_API_KEY_ATTACKER_PER_ORG="sk-..."
OPENAI_API_KEY_ATTACKER_GLOBAL="sk-..."
```

## Data

[`audit-data.zip`](audit-data.zip) contains the most important data fields from the paper's audits: the response times, prompt and completion lengths, and timestamps. When unzipped, the directory is organized as
```shell
audit-data/<provider>/<sharing_level>/<filename>.json
```

The filenames are formatted as
```shell
<provider>_<model>_<sharing_level>_n<n_samples>_p<n_prompt_tokens>_pf<prefix_fraction>_v<n_victim_requests>_<timestamp>.json
```

For example:
```shell
audit-data/openai/per_org/openai_gpt-4o-mini_per_org_n250_p5000_pf0-95_v1_2024-09-11T233151Z.json
```

Each file is structured as follows:
```jsonc
{
  "cache_hit": [
    {
      "client_time": 0.561403,  // client-side response time, in seconds
      "server_time": 0.156,  // server-side processing time, in seconds (null if not available)
      "n_prompt_tokens": 5007,  // number of prompt tokens
      "sent_timestamp": 1726097516.6008322,  // Unix timestamp of when the API request was sent
      "n_completion_tokens": 1  // number of completion tokens (only for chat models)
    },
    ...
  ],
  "cache_miss": [
    {
      // same fields as above
    },
    ...
  ],
  "victim": [
    [
      // each inner list contains the victim requests for one prompt
      {
        // same fields as above
      },
      ...
    ],
    [
      ...
    ],
    ...
  ],
  "stats": {
    // statistics, e.g., p-values, medians, means
  },
  "config": {
    // configuration parameters, e.g., model, prefix_fraction
  }
}
```

This [Google Drive folder](https://drive.google.com/drive/folders/1u3W5gFcGrOMfq6Ad8Wl0dyzsaGko2MY5?usp=drive_link) contains the full data for each API request, including the full prompts, completions/embeddings, and HTTP requests and responses. See [`clients/timing_data.py`](clients/timing_data.py#L113) for information about all data fields. The folder also contains data from the paper's ablations.

## Citation

Please cite this work using this BibTeX entry:
```bibtex
@article{gu2025auditing,
  title={Auditing Prompt Caching in Language Model APIs},
  author={Gu, Chenchen and Li, Xiang Lisa and Kuditipudi, Rohith and Liang, Percy and Hashimoto, Tatsunori},
  journal={arXiv preprint arXiv:2502.07776},
  year={2025},
  url={https://arxiv.org/abs/2502.07776},
}
```
