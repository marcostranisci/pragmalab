"""Generate two-turn content-moderation conversations from a CSV of posts.

For every post and every configured model, a first-turn question ("would you
moderate this?") is sent to the model, followed by a configured follow-up
question (e.g. "are you sure?"). One output CSV is written per
(model, follow_up_question) combination.

Usage:
    python src/generate_conversations.py --config config.yml
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class Turn:
    post_id: Any
    post_text: str
    messages: list[dict[str, str]]
    first_response: str


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


def generation_kwargs(cfg: dict) -> dict:
    run_cfg = cfg["run"]
    return {
        "max_new_tokens": run_cfg["max_new_tokens"],
        "do_sample": run_cfg["do_sample"],
        "temperature": run_cfg["temperature"],
        "top_p": run_cfg["top_p"],
    }


def generate_reply(model, tokenizer, messages: list[dict[str, str]], gen_kwargs: dict) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **gen_kwargs,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_model(model_cfg: dict):
    dtype = DTYPE_MAP[model_cfg.get("dtype", "bfloat16")]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["hf_repo_id"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_repo_id"],
        torch_dtype=dtype,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.eval()
    return tokenizer, model


def output_filename(model_name: str, first_q_id: str, follow_up_id: str, n_first_qs: int) -> str:
    if n_first_qs > 1:
        return f"{model_name}_{first_q_id}_{follow_up_id}.csv"
    return f"{model_name}_{follow_up_id}.csv"


def run(cfg: dict) -> None:
    posts = load_posts(cfg)
    gen_kwargs = generation_kwargs(cfg)
    seed = cfg["run"]["seed"]
    temperature = cfg["run"]["temperature"]
    first_questions = cfg["first_questions"]
    follow_up_questions = cfg["follow_up_questions"]

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        print(f"[{model_name}] loading {model_cfg['hf_repo_id']} ...")
        tokenizer, model = load_model(model_cfg)
        set_seed(seed)

        for fq in first_questions:
            print(f"[{model_name}] first question '{fq['id']}': generating first-turn replies "
                  f"for {len(posts)} posts ...")
            first_turns: list[Turn] = []
            for row in posts.itertuples(index=False):
                user_msg = fq["template"].format(post_text=row.post_text)
                messages = [{"role": "user", "content": user_msg}]
                first_response = generate_reply(model, tokenizer, messages, gen_kwargs)
                first_turns.append(Turn(row.post_id, row.post_text, messages, first_response))

            for fu in follow_up_questions:
                print(f"[{model_name}] follow-up '{fu['id']}': generating replies ...")
                records = []
                for turn in first_turns:
                    conversation = turn.messages + [
                        {"role": "assistant", "content": turn.first_response},
                        {"role": "user", "content": fu["template"]},
                    ]
                    follow_up_response = generate_reply(model, tokenizer, conversation, gen_kwargs)
                    records.append(
                        {
                            "post_id": turn.post_id,
                            "post_text": turn.post_text,
                            "model": model_name,
                            "seed": seed,
                            "temperature": temperature,
                            "first_question_id": fq["id"],
                            "first_question": turn.messages[0]["content"],
                            "first_response": turn.first_response,
                            "follow_up_question_id": fu["id"],
                            "follow_up_question": fu["template"],
                            "follow_up_response": follow_up_response,
                        }
                    )

                out_path = out_dir / output_filename(model_name, fq["id"], fu["id"], len(first_questions))
                pd.DataFrame.from_records(records).to_csv(out_path, index=False)
                print(f"[{model_name}] wrote {out_path}")

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
