# Stage 2: queries and candidate documents

*Status 2026-11-16: code written and tested on mock data, not yet run on real data.
Prompt selection uses two WildSEEK classifiers (request type, open-endedness).*

The benchmark pairs **queries** (a request and the answer Comma generated for it)
with **documents** from Comma's training data, judged for relevance along several
aspects (topic, style/genre, facts, textual overlap). Stage 1 labelled the corpus
(`docs/DATA.md`). Stage 2 produces the queries and, for each query, a pool of
candidate documents to be judged in stage 3.

```
WildChat-4.8M ──queries.wildchat──► prompts.parquet          factoid / analytical / creative requests, first turn
                                     │
                  queries.generate ──► generations/<run>.parquet   comma-v0.1-2t continuations = queries
                                     │
                  queries.annotate ──► annotations/{propella,weborganizer}/<run>.parquet
                                     │      (same label space as the corpus)
corpus labels (stage 1) ─────────────┤
                 queries.label_pool ──► pools/<run>.labels.<match>.parquet   label-matched sample
          (next) index retrieval   ──► pools/<run>.<system>.parquet          BM25 / InfiniSearch / dense
                                     │
                         (next) merge ► pool to judge (stage 3: LLM silver labels per aspect + human subset)
```

All outputs go below `$ATTRIB_DATA/queries/`; each comes with a
`*.manifest.json` that records how it was made (arguments, dataset/model
revision, host, counts).

## Decisions

| | Choice | Why |
|---|---|---|
| Requests | WildChat-4.8M, first user turn only | Real user requests. Later turns depend on ChatGPT's answers, which Comma never produced. |
| Strata | `info_seeking/factoid`, `info_seeking/analytical`, `creative`; no coding | Factoid answers are the clearest cases for fact attribution, analytical ones mix facts and synthesis, and creative ones are mostly about style and genre. |
| Classifiers | `tceron/info-seek-classifier` (request type, 5 classes) and `tceron/open-endedness-classifier` (Factoid / Analytical), both ModernBERT-large fine-tuned on manually labelled WildSEEK data (ODC-BY), run on each English first turn at 256 tokens as in training | Per-request labels. Reported test macro-F1 is 0.87 for request type (0.94 for info-seeking) and 0.83 for open-endedness (WildSEEK paper). |
| Cutoffs | request label probability ≥ 0.8; openness label probability ≥ 0.7 (info-seeking strata only) | Prompts are plentiful, so precision matters more than recall. Adjust after reading the label counts in the manifests. |
| Model | `common-pile/comma-v0.1-2t` | The stronger Comma (2T tokens + cool-down on high-quality sources). Same training set as 1t. |
| Prompt frame | Plain-text dialogue with two demonstrations (`queries/templates.py: FEWSHOT`) | Comma is a base model without a chat template (4,096-token context). Generation stops at the next `\nUser:`. |
| Decoding | Sampled, temperature 0.7, top-p 0.9, ≤ 768 new tokens, `--n` samples per prompt | Realistic outputs; several samples per request are possible. The stored outputs are the benchmark. |
| Document pool | Annotated subset of Comma (groups 1–2, 91.6M records; books once windowed) | Every candidate has labels; indices stay manageable. |
| Compute | cir | |

## Prompt selection (`queries.wildchat`)

Three steps, each with its own output in `queries/wildchat/`. Finished steps are
skipped on a rerun; `--redo <step>` reruns a step and the ones after it.

**filter → `candidates.parquet`** (CPU). The cheap filters run before the classifier,
so the classifier only sees unique English candidates:
1. English, not toxic, not redacted (conversation level)
2. first message is from the user, in English
3. 15–2,000 characters (about ≤ 500 tokens, leaves room in Comma's context)
4. self-contained: no URLs, no image-generator templates (Midjourney, …), no
   references to earlier turns or to attachments (`CONTEXT_PATTERNS`)
5. deduplication: exact (normalised), then by the first 80 normalised characters
   (collapses templates); the oldest conversation is kept

**classify → `classified.request.parquet`, `classified.openness.parquet`** (GPU if
available). The two classifiers run one after the other on all candidates, so
both labels are available for every prompt. Each output has the top label
(lower-cased), its probability and all class probabilities. Each classifier keeps
its own checkpoint and resumes after a kill. A finished classifier is skipped
unless the candidates changed (detected automatically) or `--redo classify` is
given. Each manifest counts every label at probability thresholds 0.5–0.95.

**sample → `prompts.parquet`** (seconds, can be rerun freely):

| stratum | request label (prob ≥ `--min-prob`) | openness label (prob ≥ `--min-openness-prob`) |
|---|---|---|
| `info_seeking/factoid` | information seeking | factoid |
| `info_seeking/analytical` | information seeking | analytical |
| `creative` | content creation | any (stored, not used) |

From each stratum, a fixed-seed random sample of `--per-stratum` (default 500) is
drawn. The manifest records how many prompts were available per stratum and the
request × openness cross-table over all candidates.

Counts after each filter go to `candidates.manifest.json`.

## Outputs

| file | one row per | columns |
|---|---|---|
| `wildchat/candidates.parquet` | unique filtered first turn | `prompt_id` (WildChat `conversation_hash`), `prompt`, `chatgpt_model`, `timestamp`, `n_turns`, `n_chars` |
| `wildchat/classified.<request\|openness>.parquet` | candidate | `prompt_id`, `label`, `prob`, `probs` (in the classifier's label order, see manifest) |
| `prompts.parquet` | sampled request | candidate columns + `stratum`, `task` (`info_seeking`/`creative`), `task_label`, `task_prob`, `openness`, `openness_prob` |
| `generations/<run>.parquet` | (request, sample) | `query_id` = `<prompt_id>:<sample>`, `prompt_id`, `sample`, `task`, `stratum`, `task_prob`, `openness`, `openness_prob`, `prompt`, `generation`, `finish_reason` (`stop`/`length`/`rejected`), `max_tokens`, `n_chars`, `empty`, `repetition` (share of repeated word 4-grams) |
| `annotations/<annotator>/<run>.parquet` | query | `query_id` + the annotator's columns, as in `docs/DATA.md` |
| `pools/<run>.labels.<match>.parquet` | (query, document) | `query_id`, `shard`, `row` [`window`, `start`, `end`], `priority`, `rank` [`text`] |
| `pools/<run>.labels.<match>.stats.parquet` | query | `query_id`, `bucket_size` (number of matching documents), `sampled`, the query's labels |

Queries with `empty` generations are not annotated or pooled. `finish_reason =
length` and high `repetition` are kept but should be looked at before judging.

## Label-matched candidates (`queries.label_pool`)

A document matches a query if the chosen single-valued labels are equal
(WebOrganizer `topic`, `format`; propella fields such as `audience_level`) and the
chosen multi-valued propella labels (`content_type`, `technical_content`, …) share
at least one value. From the matching documents, `--k` are sampled uniformly by
**bottom-k hashing**: each (query, document) pair has a fixed pseudo-random
priority and the k smallest are kept. The sample is reproducible and independent
of shard order. Samples computed on different machines merge by taking the k
smallest priorities over their union.

What this source is for, and what not:
- It finds topic/genre/register-relevant documents **without lexical overlap**,
  which retrieval systems miss. It is the main candidate source for the
  style/genre aspect.
- It is a **random** sample within the bucket. Most sampled documents will not
  be relevant for facts. Judged, it estimates the prevalence of relevant
  documents per bucket, i.e. how much an index-based pool misses (pool recall).
- `bucket_size` shows how specific the match is. Too broad (millions) → add
  fields; too narrow (< k) → drop fields.

## Running on cir

```bash
# 0. propella's corpus labels were produced on the cluster: copy them to cir first
rsync -a hpc:~/attribution_benchmark/data/annotations/propella/ data/annotations/propella/

# 1. prompts (first run downloads WildChat-4.8M; the filter step is CPU-only)
python -m queries.wildchat --stop-after filter
CUDA_VISIBLE_DEVICES=3 python -m queries.wildchat --stop-after classify   # both classifiers; check label_counts
python -m queries.wildchat --per-stratum 500                               # re-sample freely: classify is cached

# 2. generations (SGLang venv; GPUs 1-3, GPU 0 stays free)
CUDA_VISIBLE_DEVICES=3 python -m queries.generate --limit 50          # look at the preview first
CUDA_VISIBLE_DEVICES=1,2,3 python -m queries.generate --sglang-args "--dp-size 3"

# 3. labels for the generations
CUDA_VISIBLE_DEVICES=3 python -m queries.annotate propella --backend sglang
CUDA_VISIBLE_DEVICES=3 .venv-weborganizer/bin/python -m queries.annotate weborganizer

# 4. label-matched candidates (CPU)
python -m queries.label_pool --match topic,format,content_type --k 100 --with-text
```

`--run` names an output set (default `comma2t-fewshot`). A run resumes from its
checkpoint only with the same settings (`<run>.params.json`). `--limit` keeps the
checkpoint and writes `<run>.preview.parquet`.

## Next: index-based candidates

The label-matched sample needs complementing with retrieval over the text, for
the fact and overlap aspects. Every system's top-k goes into the pool
(TREC-style), so the benchmark doesn't favour one method:

- **BM25** over the 91.6M records (Lucene/Pyserini on cir), queried with the
  generation and the request + generation.
- **Exact/near-verbatim overlap**: an infini-gram index of the annotated subset,
  with InfiniSearch on top. No public infini-gram index of Comma exists; the only
  public structure is a Bloom-filter data portrait (`aalkaswan/comma-v0.1-portrait`),
  which answers membership but cannot return documents.
- **Dense** (later, P5): costly at 91.6M records; possibly only over a
  label-filtered subset.

Pool bias: InfiniSearch and steered dense retrieval will be evaluated on this
benchmark. If they also build the pool, their results are inflated. Keep the label
sample and BM25 in every pool, and report results per pool source.

## Open questions

- Access terms of `allenai/WildChat-4.8M`.
- Neither model card reports accuracy (the paper does, see above). Spot-check
  a sample per stratum before generating, and set the cutoffs from that check.
- The open-endedness classifier is defined for information-seeking requests. Its
  labels on creative prompts are stored but carry no meaning.
- `content creation` also covers transformations of given text (summarise,
  rewrite, translate) and sexual content not flagged as toxic. Decide whether
  `creative` should be narrower.
- The same study has a high-risk-domain classifier (`tceron/high-risk-classifier`:
  health, finance, legal, …). It could be added the same way if risk domains
  should be balanced or reported.
- How sensitive are the generations to the frame? Compare `fewshot` vs `plain` on
  a sample.
- Annotate the generation alone (default) or request + answer (`--with-prompt`)?
  Short answers may be labelled differently from the web documents they draw on.
- How many queries, samples per request, and candidates per query (stage-3
  labelling budget)?
- Output quality filter beyond `empty`/`repetition` (e.g. off-topic answers).
