"""Generate second-turn follow-up replies from previously generated first replies.

This is stage 2 of the two-stage pipeline: for every model it reads back the
first-turn replies written by `generate_first_replies.py` (from
`output.first_replies_dir`), keeps only the rows whose `seed` is in
`run.seeds`, and for each configured follow-up question sends the
conversation-so-far plus the follow-up and records the model's reply. One
output CSV is written per (model, first_question, follow_up) combination,
matching the original single-script pipeline's output format.

To generate follow-ups for only one seed's first replies (e.g. seed 42,
skipping seed 85 rows in the same first-replies file), set `run.seeds: [42]`
(and drop `run.n_iterations`, or set it to 1) in the config before running.

Usage:
    python src/generate_follow_up_replies.py --config config.yml
"""

from __future__ import annotations

import argparse
import csv

import pandas as pd
from transformers import set_seed

from pipeline import ConversationModel, FOLLOW_UP_FIELDNAMES, PipelineConfig, output_filename


def run(pcfg: PipelineConfig) -> None:
    out_dir = pcfg.follow_up_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n_first_qs = len(pcfg.first_questions)

    for model_cfg in pcfg.models:
        model_name = model_cfg["name"]
        first_replies_path = pcfg.first_replies_dir / f"{model_name}_first.csv"
        if not first_replies_path.exists():
            raise FileNotFoundError(
                f"No first-turn replies found for '{model_name}' at "
                f"{first_replies_path}. Run generate_first_replies.py first."
            )
        first_replies = pd.read_csv(first_replies_path)
        first_replies = first_replies[first_replies["seed"].isin(pcfg.seeds)]
        if first_replies.empty:
            raise ValueError(
                f"No first-turn replies for '{model_name}' match run.seeds={pcfg.seeds} "
                f"in {first_replies_path}."
            )

        # Open one CSV writer per (first_question, follow_up_question) up front so
        # rows are streamed into the same output file as they're generated.
        files: dict[tuple[str, str], object] = {}
        writers: dict[tuple[str, str], csv.DictWriter] = {}
        for fq in pcfg.first_questions:
            for fu in pcfg.follow_up_questions:
                out_path = out_dir / output_filename(model_name, fq["id"], fu["id"], n_first_qs)
                f = open(out_path, "w", newline="", encoding="utf-8")
                writer = csv.DictWriter(f, fieldnames=FOLLOW_UP_FIELDNAMES)
                writer.writeheader()
                files[(fq["id"], fu["id"])] = f
                writers[(fq["id"], fu["id"])] = writer

        with ConversationModel(model_cfg, pcfg.gen_kwargs) as cm:
            for fu in pcfg.follow_up_questions:
                print(f"[{model_name}] follow-up '{fu['id']}': generating replies for "
                      f"{len(first_replies)} first-turn rows ...")
                for row in first_replies.itertuples(index=False):
                    set_seed(int(row.seed))
                    conversation = [
                        {"role": "user", "content": row.first_question},
                        {"role": "assistant", "content": row.first_response},
                        {"role": "user", "content": fu["template"]},
                    ]
                    follow_up_response = cm.generate_reply(conversation)
                    writer = writers[(row.first_question_id, fu["id"])]
                    writer.writerow(
                        {
                            "post_id": row.post_id,
                            "post_text": row.post_text,
                            "model": model_name,
                            "iteration": row.iteration,
                            "seed": row.seed,
                            "temperature": row.temperature,
                            "first_question_id": row.first_question_id,
                            "first_question": row.first_question,
                            "first_response": row.first_response,
                            "follow_up_question_id": fu["id"],
                            "follow_up_question": fu["template"],
                            "follow_up_response": follow_up_response,
                        }
                    )
                for f in files.values():
                    f.flush()

        for (fq_id, fu_id), f in files.items():
            f.close()
            print(f"[{model_name}] wrote {f.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    args = parser.parse_args()
    pcfg = PipelineConfig(args.config)
    run(pcfg)


if __name__ == "__main__":
    main()
