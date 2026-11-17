"""Stage 2 of the attribution benchmark: queries and candidate documents.

    queries.wildchat     WildChat prompts -> filtered, deduplicated, sampled prompts
    queries.generate     prompts -> Comma continuations (the benchmark queries)
    queries.annotate     generations -> propella / WebOrganizer labels (same space as the corpus)
    queries.label_pool   generations + corpus labels -> label-matched candidate documents

Everything is written below $ATTRIB_DATA/queries (see pipeline/common.py).
"""
