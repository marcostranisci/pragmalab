"""Smoke-test the two-stage generation pipeline against a single small (0.5B) model.

Runs both stages back to back with config_test.yml, which pins a 0.5B model
and a handful of iterations/seeds distinct from the main config.yml, so the
pipeline can be sanity-checked quickly (e.g. on a laptop or before launching
a full multi-model run on the HPC).

Usage:
    python src/test_small_model.py
    python src/test_small_model.py --config config_test.yml
"""

from __future__ import annotations

import argparse

import generate_first_replies
import generate_follow_up_replies
from pipeline import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_test.yml", help="Path to test config")
    args = parser.parse_args()
    pcfg = PipelineConfig(args.config)
    generate_first_replies.run(pcfg)
    generate_follow_up_replies.run(pcfg)


if __name__ == "__main__":
    main()
