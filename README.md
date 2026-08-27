# Moderation conversation generator

Generates two-turn conversations about content moderation from a CSV of posts:

1. First turn: "I saw this post: ... Would you moderate it?"
2. Follow-up turn: "Are you sure?" (or whatever is configured)

Everything — input CSV, models, generation parameters and prompt wording — is
controlled from `config.yml`. Each post can be generated multiple times
(different seeds) per `(model, follow_up_question)` pair — see
`run.n_iterations` / `run.seeds` below.

The pipeline is split into two stages, each its own script, so a long run
across several models doesn't have to fit inside a single job's time limit:

1. `src/generate_first_replies.py` generates every first-turn ("would you
   moderate this?") reply and writes one CSV per model to
   `output.first_replies_dir` (default `output/first_replies/`), e.g.
   `ministral_first.csv`.
2. `src/generate_follow_up_replies.py` reads those CSVs back in and, for
   each configured follow-up question, generates the second-turn reply.
   One output CSV is written per `(model, follow_up_question)` combination,
   e.g. `ministral_are_you_sure.csv`, streamed out with `csv.DictWriter` as
   generations complete.

Both scripts share their config/data loading and model-loading code via
`src/pipeline.py` (`PipelineConfig` and `ConversationModel`).

## Setup (HPC, conda, Python 3.12)

```bash
conda env create -f environment.yml
conda activate modconv

# Install PyTorch matching your cluster's CUDA toolkit, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

If you're pulling gated models (e.g. Llama), authenticate first:

```bash
huggingface-cli login
```

## Configure

Edit `config.yml`:

- `input.csv_path`, `input.post_id_column`, `input.post_text_column`: point these
  at your real CSV and its column names.
- `models`: list of Hugging Face repo IDs to load with
  `AutoModelForCausalLM`/`AutoTokenizer`. Add/remove entries freely.
- `first_questions` / `follow_up_questions`: prompt templates. `{post_text}` is
  substituted into `first_questions` templates. Add more entries to try several
  phrasings — output filenames disambiguate automatically when there is more
  than one first-question variant.
- `run.seed`, `run.temperature`, `run.top_p`, `run.max_new_tokens`: generation
  settings, applied identically across all models for comparability.
- `run.n_iterations` / `run.seeds`: repeat generation multiple times per
  `(model, follow_up_question, post)` to sample several generations for the
  same item.
  - Set `run.seeds: [42, 43, 44]` to run exactly those seeds (3 iterations).
  - Set `run.n_iterations: 3` with `run.seed: 42` to auto-derive seeds
    `[42, 43, 44]`.
  - Leave both unset for a single iteration using `run.seed` (default,
    backward-compatible with older configs).
  - If both are set, `n_iterations` must match `len(seeds)`.
  - Each output row carries `iteration` (1-indexed) and `seed` columns.
- `run.limit`: set to a small integer while testing; `null` to process every row.

## Run

Run the two stages in order:

```bash
python src/generate_first_replies.py --config config.yml
python src/generate_follow_up_replies.py --config config.yml
```

On the HPC, submit them as two separate jobs so each stays well inside its
time limit — `pragmalab_first_replies.job` and
`pragmalab_follow_up_replies.job`. Chain them so the second only starts once
the first succeeds:

```bash
first_id=$(sbatch --parsable pragmalab_first_replies.job)
sbatch --dependency=afterok:$first_id pragmalab_follow_up_replies.job
```

Stage 1 writes one CSV per model to `output.first_replies_dir` (default
`output/first_replies/`), each containing: `post_id`, `post_text`, `model`,
`iteration`, `seed`, `temperature`, `first_question_id`, `first_question`,
`first_response`.

Stage 2 reads those back in and writes to `output.dir` (default `outputs/`),
one CSV per `(model, follow_up_question)` pair, adding
`follow_up_question_id`, `follow_up_question` and `follow_up_response` to
the same columns. When `run.n_iterations` (or `run.seeds`) is greater than
1, files contain one row per `(post, iteration)`, distinguished by the
`iteration`/`seed` columns.

## Smoke-testing on a small model

`src/test_small_model.py` runs both stages back to back against a single
small (`Qwen2.5-0.5B-Instruct`) model, configured via `config_test.yml` with
its own small `run.n_iterations`/`run.seeds` distinct from the main
`config.yml`. Use it for a fast local sanity check before launching a full
multi-model run on the HPC:

```bash
python src/test_small_model.py
# or point it at a different config:
python src/test_small_model.py --config config_test.yml
```

`config_test.yml` also sets `run.limit: 2` and a short `max_new_tokens` so
the whole thing runs in seconds. A small `data/example_posts.csv` is
included for this purpose.
