# BM25 vs. MPNet (zero-shot bi-encoder): query-by-query comparison

Source data: `results/bm25_dev.json`, `results/bi_encoder_dev.json` (English development split, 10 queries).

## Setup note: Recall@10 ceiling

Relevant-doc pool sizes range from 8 to 112 across queries. Recall@10 is capped at
`min(10, |R|) / |R|`, so for a query with 112 relevant résumés, recall@10 can never
exceed 0.089 even with a perfect top-10. Checking actual hits against that ceiling
shows **7 of 10 queries have identical, perfect top-10 precision for both models**
(`29243`, `35129`, `37020`, `38671`, `39060`, `44719`, `47576`) — every retrieved doc
in the top 10 is relevant for both models. The real differences live in the
remaining 3 queries.

| qid | \|R\| | ceiling@10 | BM25 hits/10 | MPNet hits/10 |
|---|---|---|---|---|
| 29243 | 44 | 0.227 | 10 | 10 |
| 32447 | 32 | 0.312 | 8 | 7 |
| 35129 | 8 | 1.000 | 8 | 8 |
| 36044 | 56 | 0.179 | 8 | 10 |
| 37020 | 44 | 0.227 | 10 | 10 |
| 38671 | 40 | 0.250 | 10 | 10 |
| 39060 | 112 | 0.089 | 10 | 10 |
| 44719 | 72 | 0.139 | 10 | 10 |
| 46795 | 32 | 0.312 | 4 | 8 |
| 47576 | 32 | 0.312 | 10 | 10 |

## MPNet clearly wins — query `46795` "Paralegal, Privacy and Information Management"

| | Recall@10 | Recall@50 | NDCG@10 | MRR |
|---|---|---|---|---|
| BM25 | 0.125 (4/32) | 0.844 | 0.293 | 0.200 |
| MPNet | 0.250 (8/32) | **1.000** (finds all 32) | 0.851 | **1.000** |

BM25's top hit is doc `9130`, "Director, Compliance & Ethics" — irrelevant, but
scores high because it shares surface keywords (compliance, privacy, regulatory).
Ranks 1-4 are all irrelevant this way; the first true hit doesn't appear until rank
5 (`7094`, whose email is literally `amir.hassan.paralegal@gmail.com`).

MPNet's top hit is doc `14060`: "Detail-oriented Corporate Compliance Paralegal
with extensive experience supporting legal and compliance teams..." — no exact
phrase overlap with the query title, but the embedding correctly recognizes the
semantic match. BM25 gets fooled by generic compliance/privacy vocabulary shared
across many non-paralegal roles, while MPNet distinguishes "paralegal" from
"compliance director" semantically.

`36044` "Failure Analysis Engineer" shows the same pattern more mildly: MPNet gets
a perfect 10/10 in top-10 vs BM25's 8/10 (NDCG@10 1.000 vs 0.852).

## BM25 clearly wins — query `32447` "Senior Mechanical/HVAC Engineer"

| | Recall@10 | Recall@50 |
|---|---|---|
| BM25 | 0.250 (8/10 hits) | **0.875** (28/32) |
| MPNet | 0.219 (7/10 hits) | 0.781 (25/32) |

Both models get pulled toward "Facilities" résumés that mention HVAC only as a
peripheral skill rather than as the job itself — e.g. doc `3930`, "Director -
Facilities Engineering and Construction," is a false positive for both. MPNet
pulls in two more such false positives that BM25 correctly excludes: `21188`
(another Facilities Engineering director) and `18934` ("facilities professional...
overseeing HVAC systems" — HVAC is a bullet point, not the role). BM25's
exact-term matching on "Mechanical" / "HVAC" / "Engineer" is more discriminating
here than semantic similarity, which conflates "facilities management with HVAC
exposure" and "HVAC/mechanical engineering design" as topically close.

## Both fail — the long tail of `36044` "Failure Analysis Engineer"

Even though both models get perfect top-10 precision on this query, 16 of the 56
relevant résumés never appear in either model's top 50 (BM25 recall@50 = 0.536,
MPNet = 0.571 — both roughly half of the ceiling of 0.893 for this pool size).
Example of a missed-by-both résumé (`12220`): "Accomplished Staff Engineer with
12+ years... embedded systems design, FPGA development, hardware-software
co-design... signal processing, control systems." It's labeled relevant to
"Failure Analysis Engineer," but shares no failure-analysis vocabulary for BM25
to latch onto, and its embedding doesn't land anywhere near the query's semantic
neighborhood either. This is a genuine shared blind spot — the taxonomy's notion
of "relevant" is broader than what either lexical or semantic surface-level
matching can capture.
