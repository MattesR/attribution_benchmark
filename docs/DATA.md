# Comma v0.1 silver annotations — data card

*Status as of 2026-10-15. Repo: github.com/MattesR/attribution_benchmark
(pipeline code + `README.md` for running it).*

## Purpose

Stage 1 of the attribution benchmark (P4 of the thesis). Every document of the
Comma v0.1 training corpus gets automatic (silver) labels. Later stages use them
to **pre-filter candidate documents** for attribution, before defining relevance
aspects (topic, style, facts) and producing LLM silver labels for
query–document pairs. Annotating the full corpus is itself a planned
contribution: a pretraining corpus enriched with metadata.

## Corpus

- `common-pile/comma_v0.1_training_dataset` on Hugging Face, **pinned at revision
  `5afc546db324e7f39f297ba757c9a60547151e7c`**. ~784M documents, 31 sources,
  each split into 64 gzipped JSONL files (`<source>/<source>.chunk.NN.jsonl.gz`).
  Each record has **only `text`**: no ID, no URL, no source metadata.
- The source is known from the file path. Per-source token counts and repeat
  rates (how often Comma saw each source) are in the dataset card.
- `wikiteam/` additionally has 64 files `NNNNN_wikiteam.json.gz` with the same
  format. It's unclear whether they belong to the training mix. They are
  currently included (open question).

## Scope and order

1. Web crawl: `cccc`, `news`, `foodista`, `public_domain_review` (in progress)
2. Web-like: `wikimedia`, `wikiteam`, `stackexchange`, `libretexts`, `pressbooks`, `oercommons`
3. Non-web sources (papers, patents, legal, books, code, …): later or not at
   all, since they already carry metadata and WebOrganizer's web taxonomy
   fits them poorly.

## Document identity and joining text

A document is **`(shard, row)`**: `shard` is the repo file path (e.g.
`cccc/cccc.chunk.00.jsonl.gz`), `row` is the 0-based line number in it. Stable
because the revision is pinned. Annotation files do **not** contain the text;
join it from the raw file:

```python
import gzip, json, pyarrow.parquet as pq
shard = "cccc/cccc.chunk.00"
texts = [json.loads(l)["text"] for l in gzip.open(f"data/raw/{shard}.jsonl.gz", "rt")]
ann = pq.read_table(f"data/annotations/propella/{shard}.parquet").to_pandas()
ann["text"] = ann["row"].map(texts.__getitem__)
```

## Files

Per shard and annotator, one Parquet file with one row per document, in file order:

```
data/raw/<source>/<file>.jsonl.gz                       original text (kept)
data/annotations/weborganizer/<source>/<file>.parquet
data/annotations/propella/<source>/<file>.parquet
progress.tsv                                            one line per downloaded / annotated shard
```

Data lives on the machines that produced it (ITU HPC home directory, and the
external machine `cir`); there is no central copy yet. `progress.tsv` (in git)
lists every finished shard with host, doc count, failures and duration.

## Annotator 1: WebOrganizer (Wettig et al., ICML 2025)

`WebOrganizer/TopicClassifier-NoURL` and `FormatClassifier-NoURL`
(gte-base-en-v1.5, 140M). These are the NoURL variants, because Comma has no URLs.
Input truncated at 8192 tokens, bf16, as in WebOrganizer's own scripts.

| column | type | meaning |
|---|---|---|
| `row` | int | line in the shard |
| `topic`, `format` | str | argmax label |
| `topic_prob`, `format_prob` | float32 | its probability |
| `topic_probs`, `format_probs` | list[float32], len 24 | full distribution, in label order below |

- **Topics:** Adult, Art & Design, Software Dev., Crime & Law, Education & Jobs,
  Hardware, Entertainment, Social Life, Fashion & Beauty, Finance & Business, Food &
  Dining, Games, Health, History, Home & Hobbies, Industrial, Literature, Politics,
  Religion, Science & Tech., Software, Sports & Fitness, Transportation, Travel
- **Formats:** Academic Writing, Content Listing, Creative Writing, Customer
  Support, Comment Section, FAQ, Truncated, Knowledge Article, Legal Notices,
  Listicle, News Article, Nonfiction Writing, About (Org.), News (Org.), About
  (Pers.), Personal Blog, Product Page, Q&A Forum, Spam / Ads, Structured Data,
  Documentation, Audio Transcript, Tutorial, User Review

Status: tested; corpus-wide run not started yet.

## Annotator 2: propella-1 (ellamind, arXiv 2602.12414)

`ellamind/propella-1-4b` (revision `bf607e62`), served with SGLang, JSON schema
enforced. **Decoding is sampled, not greedy:** the pipeline sets no
temperature, so SGLang applies the model's `generation_config.json` defaults
(temperature 0.7, top-k 20, top-p 0.8). This is confirmed in the server log
(`sampling_defaults='model'`), and these are the Qwen3 base model's defaults,
not a propella-specific choice. Each server start draws a new random seed, so
labels are not exactly reproducible (open question, see below). Input truncated at 50,000 characters (model card
recommendation). Prompt and schema come from the model repo's `propella.py`;
the allowed values for every property are in its `property_descriptions.md`.

| column | type |
|---|---|
| `row` | int |
| `one_sentence_description` | str: ~10-word English summary |
| `content_integrity`, `content_ratio`, `content_length`, `content_quality`, `information_density`, `educational_value`, `reasoning_indicators`, `audience_level`, `commercial_bias`, `time_sensitivity`, `content_safety`, `pii_presence` | str, one enum value each |
| `content_type`, `business_sector`, `technical_content`, `regional_relevance`, `country_relevance` | list[str], multi-select |
| `ok` | bool: output parsed and schema-valid (rows with `ok=False` have nulls) |

Status: **running** on `cccc` (cluster: 2× H100 on cn11 + L40S on cn19;
`cir`: A100s). Throughput on short documents so far: ~42 docs/s per H100,
~32 per L40S, ~12 per A100. Full-file numbers are pending.

## Caveats for downstream use

- **propella decoding is sampled** (see above): re-annotating a document can
  give slightly different labels.
- **Precision differs by machine:** cluster GPUs run propella in fp8, `cir`
  (A100) in bf16 with Triton attention. The model card reports equal quality,
  but labels are not bit-identical across machines. `progress.tsv` records
  which host annotated each shard.
- **Silver labels:** propella-4B agrees with a Gemini reference at ~0.78
  (overall score from its model card). WebOrganizer is distilled from Llama-405B
  labels. Neither has been validated on Comma yet.
- **WebOrganizer `Truncated`** is a real format label (fragment-like pages),
  not a processing artifact. It's frequent in `news`.
- **Long documents:** propella sees only the first 50k characters, WebOrganizer
  only the first 8192 tokens.
- **Comma upweights sources differently** (repeat rates). This doesn't affect
  the labels, but it matters if annotations are used to reason about what the
  model saw how often.

## Open questions

- propella decoding: switch to greedy (temperature 0) for reproducible labels?
- `wikiteam/NNNNN_wikiteam.json.gz`: part of the training mix or not?
- Central storage and release of the annotations (Hugging Face dataset?).
- Validation of the silver labels on a Comma sample.
