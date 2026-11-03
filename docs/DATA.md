# Comma v0.1 silver annotations — data card

*Status as of 2026-11-03. Repo: github.com/MattesR/attribution_benchmark
(pipeline code + `README.md` for running it).*

## Purpose

Stage 1 of the attribution benchmark (P4 of the thesis). Every document of the
Comma v0.1 training corpus gets automatic (silver) labels. Later stages use them
to **pre-filter candidate documents** for attribution, before defining relevance
aspects (topic, style, facts) and producing LLM silver labels for
query–document pairs. Annotating the full corpus is itself a planned
contribution: a pretraining corpus enriched with metadata.

What the labels can and cannot do: they describe **topic, genre/format, style
and register** (audience, reasoning, quality) of a whole record or window. They
do **not** say whether a text contains a specific fact or wording; that needs
retrieval over the text itself (lexical, dense, or exact match).

## Corpus

- `common-pile/comma_v0.1_training_dataset` on Hugging Face, **pinned at revision
  `5afc546db324e7f39f297ba757c9a60547151e7c`**. ~784M records, 31 sources,
  each split into 64 gzipped JSONL files (`<source>/<source>.chunk.NN.jsonl.gz`).
  Each record has **only `text`**: no ID, no URL, no source metadata.
- **Comma caps records at 128,000 characters.** Longer texts (books, long
  papers) are cut into several records, blindly (Gutenberg: often mid-word) and
  scattered across files. A record is the unit the model was trained on.
- The source is known from the file path. Per-source token counts and repeat
  rates (how often Comma saw each source) are in the dataset card.
- `wikiteam/` additionally has 64 files `NNNNN_wikiteam.json.gz` with the same
  format. It's unclear whether they belong to the training mix; they are
  annotated (open question).
- Comma's PII scrubbing produces artifacts, e.g. enzyme numbers like
  `EC 1.11.1.6` replaced by `<IP_ADDRESS>` in scientific text.

## Scope and status

| Group | Sources | Unit | propella | WebOrganizer |
|---|---|---|---|---|
| 1. Web crawl | `cccc`, `news`, `foodista`, `public_domain_review` | record | **done** | **done** |
| 2. Web-like | `wikimedia`, `wikiteam`, `stackexchange`, `libretexts`, `pressbooks`, `oercommons` | record | **done** | **done** |
| 3a. Books | `project_gutenberg`, `doab` | window (≤32k chars) | next | next |
| 3b. Other non-web | papers, legal, code, transcripts, … | open | open | open |

Groups 1 and 2: **704 files, 91,609,462 records**, about 34% of Comma's
effective training tokens (tokens × repeat rate). Patents (`uspto`) are not
planned for label-based annotation: attribution there is mostly verbatim
overlap, for which an exact-match index (OLMoTrace-style, infini-gram) is the
better fallback.

## Document identity and joining text

A record is **`(shard, row)`**: `shard` is the repo file path (e.g.
`cccc/cccc.chunk.00.jsonl.gz`), `row` is the 0-based line number in it. Stable
because the revision is pinned. Window annotations add **`window, start, end`**:
the window's number and its exact character range `[start, end)` in the
record. Annotation files do **not** contain the text; join it from the raw file:

```python
import gzip, json, pyarrow.parquet as pq
shard = "cccc/cccc.chunk.00"
texts = [json.loads(l)["text"] for l in gzip.open(f"data/raw/{shard}.jsonl.gz", "rt")]
ann = pq.read_table(f"data/annotations/propella/{shard}.parquet").to_pandas()
ann["text"] = ann["row"].map(texts.__getitem__)
# windows: texts[row][start:end]
```

## Files

One Parquet file per shard and annotator, one row per record (or window), in order:

```
data/raw/<source>/<file>.jsonl.gz                          original text (kept)
data/annotations/weborganizer/<source>/<file>.parquet      record level
data/annotations/propella/<source>/<file>.parquet
data/annotations/weborganizer-w32000/<source>/<file>.parquet  window level (books)
data/annotations/propella-w32000/<source>/<file>.parquet
progress.tsv                                               one line per downloaded / annotated shard
```

Data lives on the machines that produced it: propella on the ITU HPC home
directory, WebOrganizer (groups 1–2) on `cir`. There is no central copy yet.
`progress.tsv` (in git) lists every finished shard with host, unit count,
failures and duration.

## Annotator 1: WebOrganizer (Wettig et al., ICML 2025)

`WebOrganizer/TopicClassifier-NoURL` and `FormatClassifier-NoURL`
(gte-base-en-v1.5, 140M). These are the NoURL variants, because Comma has no URLs.
Input truncated at 8192 tokens, bf16, as in WebOrganizer's own scripts. Needs
`transformers<5` (its model code breaks on 5.x).

| column | type | meaning |
|---|---|---|
| `row` (+ `window`, `start`, `end`) | int | record (and window) |
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

## Annotator 2: propella-1 (ellamind, arXiv 2602.12414)

`ellamind/propella-1-4b` (revision `bf607e62`), served with SGLang, JSON schema
enforced, output limited to 640 tokens. **Decoding is sampled, not greedy:**
the pipeline sets no temperature, so SGLang applies the model's
`generation_config.json` defaults (temperature 0.7, top-k 20, top-p 0.8). This
is confirmed in the server log (`sampling_defaults='model'`); these are the
Qwen3 base model's defaults, not a propella-specific choice. Each server start
draws a new random seed, so labels are not exactly reproducible. Input is
truncated at 50,000 characters (model card recommendation) and, for
token-dense text (symbols, emoji, some scripts), at ~62,800 tokens, so that
prompt and answer fit the 65,536-token context; both end with
`<truncated_content>`. Prompt and schema come from the model repo's
`propella.py`; the allowed values for every property are in its
`property_descriptions.md`.

| column | type |
|---|---|
| `row` (+ `window`, `start`, `end`) | int |
| `one_sentence_description` | str: ~10-word English summary |
| `content_integrity`, `content_ratio`, `content_length`, `content_quality`, `information_density`, `educational_value`, `reasoning_indicators`, `audience_level`, `commercial_bias`, `time_sensitivity`, `content_safety`, `pii_presence` | str, one enum value each |
| `content_type`, `business_sector`, `technical_content`, `regional_relevance`, `country_relevance` | list[str], multi-select |
| `ok` | bool: output parsed and schema-valid (rows with `ok=False` have nulls) |

`ok=False` is rare (e.g. 117 of 484,184 records in one stackexchange file,
0.02%): outputs cut at the 640-token limit, or documents the server rejected.
All propella annotations of groups 1–2 were produced on the cluster
(H100/L40S, fp8).

## Truncation: what the record-level labels saw

Record-level labels describe only the start of long records: propella reads
50,000 characters, WebOrganizer 8,192 tokens (~32,000 characters). This
matches both annotators' published usage. Measured on one file per source:

| Source | Median chars | Records > 50k | Text unseen by propella | Records > 32k | Text unseen by WebOrganizer |
|---|---|---|---|---|---|
| cccc | 2,077 | 4.1% | 8.8% | 9.2% | 22.6% |
| wikimedia | 1,168 | 1.0% | 10.9% | 1.7% | 17.5% |
| wikiteam (both file types) | 201 / 491 | 0.0–0.1% | ~2% | 0.0–0.1% | 3–4% |
| stackexchange | 2,160 | 0.0% | 0.1% | 0.1% | 0.4% |
| pressbooks | 6,044 | 2.8% | 7.5% | 6.7% | 14.2% |
| libretexts | 4,613 | 1.3% | 5.0% | 2.4% | 8.6% |
| oercommons | 4,653 | 6.0% | 19.8% | 8.3% | 29.6% |
| news, foodista, public_domain_review | < 3,300 | 0% | 0% | ≤ 0.1% | ≤ 0.3% |

Overall about **0.5% of the 91.6M records** (~450k, mostly `cccc` and
`wikimedia`) are truncated for propella; because these are the longest records,
roughly 5–6% of the text (WebOrganizer: ~10%) is unseen. For web pages and
articles the first 50k characters usually determine topic and type, so this was
not rerun. Planned: a **`truncated` flag** per record and annotator, computed
from the raw text length when the annotations are merged. If needed later, the
affected records can be annotated additionally in window mode.

## Window mode (books)

For books, record-level labels would describe little: Gutenberg records are
blind 128k-character pieces (87% exceed 50k characters, 48% start
mid-sentence). `--windows 32000` splits every record longer than 32,000
characters into a fixed number of balanced windows (`ceil(len / 28,800)`, median
~25,500 characters), cut at the nearest heading (Markdown or `CHAPTER`), else
paragraph break, else line or sentence end. Records up to 32,000 characters
stay one window. Both annotators read every window in full.

| Source | Records | Records > 32k | Windows (est.) | Cuts at headings |
|---|---|---|---|---|
| project_gutenberg | 185,133 | 92% | ~790k | 23% (rest: paragraphs) |
| doab | 461,558 | 22% | ~760k | 55% (rest: paragraphs) |

Caveat: propella's `content_integrity` and `content_length` judge the window,
so many windows are rated "fragment" / shorter than the book they come from.

## Caveats for downstream use

- **propella decoding is sampled** (see above): re-annotating a document can
  give slightly different labels.
- **Precision differs by machine:** propella ran in fp8 on H100/L40S;
  WebOrganizer in bf16. Annotations from different machines are not
  bit-identical; `progress.tsv` records the host per shard.
- **Silver labels:** propella-4B agrees with a Gemini reference at ~0.78
  (overall score from its model card). WebOrganizer is distilled from Llama-405B
  labels. Neither has been validated on Comma yet.
- **WebOrganizer `Truncated`** is a real format label (fragment-like pages),
  not a processing artifact. It's frequent in `news`.
- **Long records** are labeled by their start (see Truncation).
- **Comma upweights sources differently** (repeat rates). This doesn't affect
  the labels, but it matters if annotations are used to reason about what the
  model saw how often.

## Open questions

- propella decoding: greedy (temperature 0) for reproducible labels?
- `wikiteam/NNNNN_wikiteam.json.gz`: part of the training mix or not?
- Central storage and release of the annotations (Hugging Face dataset?).
- Validation of the silver labels on a Comma sample.
- Further non-web sources (papers, legal, code); exact-match fallback for patents.
