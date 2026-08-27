"""Shared config/data loading and model wrapper for the moderation-conversation
generator.

This module holds everything that both generation stages need in common:
config loading, the posts dataframe, seed/generation-kwarg resolution, and a
`ConversationModel` class that wraps a Hugging Face tokenizer + causal LM.
The two stage scripts (`generate_first_replies.py` and
`generate_follow_up_replies.py`) each import from here instead of duplicating
this logic.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Columns written by generate_first_replies.py, and read back in by
# generate_follow_up_replies.py.
FIRST_REPLY_FIELDNAMES = [
    "post_id",
    "post_text",
    "model",
    "iteration",
    "seed",
    "temperature",
    "first_question_id",
    "first_question",
    "first_response",
]

# Columns written by generate_follow_up_replies.py.
FOLLOW_UP_FIELDNAMES = FIRST_REPLY_FIELDNAMES + [
    "follow_up_question_id",
    "follow_up_question",
    "follow_up_response",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_posts(cfg: dict) -> pd.DataFrame:
    input_cfg = cfg["input"]
    df = pd.read_csv(input_cfg["csv_path"])
    id_col = input_cfg["post_id_column"]
    text_col = input_cfg["post_text_column"]
    if id_col not in df.columns or text_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{id_col}' and '{text_col}'; "
            f"found {list(df.columns)}"
        )
    limit = cfg["run"].get("limit")
    if limit:
        df = df.head(int(limit))
    return df[[id_col, text_col]].rename(columns={id_col: "post_id", text_col: "post_text"})


def resolve_seeds(run_cfg: dict) -> list[int]:
    """Work out the list of seeds to iterate over for each (model, question).

    - `run.seeds: [1, 2, 3]` runs exactly those seeds (3 iterations).
    - `run.n_iterations: 3` with `run.seed: 42` runs seeds [42, 43, 44].
    - Neither set: a single iteration using `run.seed`.
    - Both set: `n_iterations` must match `len(seeds)`.
    """
    seeds = run_cfg.get("seeds")
    n_iterations = run_cfg.get("n_iterations")
    if seeds is not None:
        seeds = [int(s) for s in seeds]
        if n_iterations is not None and int(n_iterations) != len(seeds):
            raise ValueError(
                f"run.n_iterations ({n_iterations}) does not match the number "
                f"of run.seeds ({len(seeds)})"
            )
        return seeds
    base_seed = int(run_cfg["seed"])
    n_iterations = int(n_iterations) if n_iterations else 1
    return [base_seed + i for i in range(n_iterations)]


def generation_kwargs(cfg: dict) -> dict:
    run_cfg = cfg["run"]
    return {
        "max_new_tokens": run_cfg["max_new_tokens"],
        "do_sample": run_cfg["do_sample"],
        "temperature": run_cfg["temperature"],
        "top_p": run_cfg["top_p"],
    }


def output_filename(model_name: str, first_q_id: str, follow_up_id: str, n_first_qs: int) -> str:
    if n_first_qs > 1:
        return f"{model_name}_{first_q_id}_{follow_up_id}.csv"
    return f"{model_name}_{follow_up_id}.csv"


class PipelineConfig:
    """Loads config.yml and derives everything both stages need from it."""

    def __init__(self, path: str):
        self.cfg = load_config(path)
        self.posts = load_posts(self.cfg)
        self.gen_kwargs = generation_kwargs(self.cfg)
        self.seeds = resolve_seeds(self.cfg["run"])
        self.temperature = self.cfg["run"]["temperature"]
        self.first_questions = self.cfg["first_questions"]
        self.follow_up_questions = self.cfg.get("follow_up_questions", [])
        self.models = self.cfg["models"]

    @property
    def first_replies_dir(self) -> Path:
        return Path(self.cfg["output"].get("first_replies_dir", "output/first_replies"))

    @property
    def follow_up_dir(self) -> Path:
        return Path(self.cfg["output"]["dir"])


class ConversationModel:
    """Wraps a Hugging Face tokenizer + causal LM for chat-style generation.

    Used as a context manager so the model is always freed afterwards:

        with ConversationModel(model_cfg, gen_kwargs) as cm:
            reply = cm.generate_reply(messages)
    """

    def __init__(self, model_cfg: dict, gen_kwargs: dict):
        self.name = model_cfg["name"]
        self.model_cfg = model_cfg
        self.gen_kwargs = gen_kwargs
        self.tokenizer = None
        self.model = None

    def __enter__(self) -> "ConversationModel":
        self.load()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.unload()

    def load(self) -> None:
        dtype = DTYPE_MAP[self.model_cfg.get("dtype", "bfloat16")]
        trust_remote_code = self.model_cfg.get("trust_remote_code", False)
        print(f"[{self.name}] loading {self.model_cfg['hf_repo_id']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg["hf_repo_id"],
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg["hf_repo_id"],
            torch_dtype=dtype,
            device_map=self.model_cfg.get("device_map", "auto"),
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def unload(self) -> None:
        del self.model, self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_reply(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **self.gen_kwargs,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
