"""Generate first-turn ("would you moderate this?") replies for every post.

This is stage 1 of the two-stage pipeline: it only produces the first-turn
replies for every (model, first_question, post, iteration) combination and
writes one CSV per model to `output.first_replies_dir`. Run
`generate_follow_up_replies.py` afterwards to produce the second-turn
replies from these files.

Splitting the pipeline this way lets each stage finish inside a single job's
time limit instead of one long-running job that does both turns for every
model.

Usage:
    python src/generate_first_replies.py --config config.yml
"""

from __future__ import annotations

import argparse
import csv

from transformers import set_seed

from pipeline import ConversationModel, FIRST_REPLY_FIELDNAMES, PipelineConfig


def run(pcfg: PipelineConfig) -> None:
    out_dir = pcfg.first_replies_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in pcfg.models:
        model_name = model_cfg["name"]
        out_path = out_dir / f"{model_name}_first.csv"

        with open(out_path, "w", newline="", encoding="utf-8") as f, ConversationModel(
            model_cfg, pcfg.gen_kwargs
        ) as cm:
            writer = csv.DictWriter(f, fieldnames=FIRST_REPLY_FIELDNAMES)
            writer.writeheader()

            for iteration, seed in enumerate(pcfg.seeds, start=1):
                print(f"[{model_name}] iteration {iteration}/{len(pcfg.seeds)} (seed={seed})")
                set_seed(seed)

                for fq in pcfg.first_questions:
                    print(
                        f"[{model_name}] first question '{fq['id']}': generating first-turn "
                        f"replies for {len(pcfg.posts)} posts ..."
                    )
                    for row in pcfg.posts.itertuples(index=False):
                        user_msg = fq["template"].format(post_text=row.post_text)
                        messages = [{"role": "user", "content": user_msg}]
                        first_response = cm.generate_reply(messages)
                        writer.writerow(
                            {
                                "post_id": row.post_id,
                                "post_text": row.post_text,
                                "model": model_name,
                                "iteration": iteration,
                                "seed": seed,
                                "temperature": pcfg.temperature,
                                "first_question_id": fq["id"],
                                "first_question": user_msg,
                                "first_response": first_response,
                            }
                        )
                    f.flush()

        print(f"[{model_name}] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    args = parser.parse_args()
    pcfg = PipelineConfig(args.config)
    run(pcfg)


if __name__ == "__main__":
    main()
