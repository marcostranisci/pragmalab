# Moderation conversation generator

Generates two-turn conversations about content moderation from a CSV of posts:

1. First turn: "I saw this post: ... Would you moderate it?"
2. Follow-up turn: "Are you sure?" (or whatever is configured)

Everything — input CSV, models, generation parameters and prompt wording — is
controlled from `config.yml`. One output CSV is written per
`(model, follow_up_question)` combination, e.g. `ministral_are_you_sure.csv`.

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
- `run.limit`: set to a small integer while testing; `null` to process every row.

## Run

```bash
python src/generate_conversations.py --config config.yml
```

Output CSVs land in `outputs/` (configurable via `output.dir`), one per
`(model, follow_up_question)` pair, each containing: `post_id`, `post_text`,
`model`, `seed`, `temperature`, `first_question_id`, `first_question`,
`first_response`, `follow_up_question_id`, `follow_up_question`,
`follow_up_response`.

A small `data/example_posts.csv` is included for a smoke test — set
`run.limit: 2` and try a single small model first before launching the full
run on the HPC.
