# talentclef-ranking

Retrieval baselines and bi-encoder fine-tuning experiments for the [TalentCLEF](https://talentclef.github.io/talentclef/) Task A job-to-resume ranking benchmark (English development split).

## Project Structure

```
talentclef-ranking/
├── data/         # TalentCLEF Task A data (not committed - see Data below)
├── src/          # core library code (BM25, bi-encoder, evaluation metrics, data loading)
├── scripts/      # runnable entry-point scripts
├── results/      # model outputs, metrics, artifacts
├── notebooks/    # exploratory notebooks
├── tests/        # unit and integration tests
├── README.md
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Expects the TalentCLEF Task A English development split (queries, corpus, qrels) under `data/TaskA/development/en/`. The dataset is excluded from version control (`.gitignore`) since it's large and subject to redistribution restrictions — download it separately from the TalentCLEF shared task and place it at that path.

## Baselines

**BM25** (`src/bm25.py`, no external dependencies):

```bash
python scripts/evaluate_bm25.py
```

**Zero-shot bi-encoder** (`sentence-transformers/all-mpnet-base-v2`, `src/bi_encoder.py`):

```bash
python scripts/evaluate_bi_encoder.py
```

Both report Recall@10, Recall@50, MRR, and NDCG@10 (`src/evaluation.py`) over all 10 development queries and save to `results/bm25_dev.json` / `results/bi_encoder_dev.json`.

## Bi-Encoder Fine-Tuning with Hard Negatives

`all-mpnet-base-v2` was fine-tuned as a bi-encoder for job-to-resume retrieval using triplet loss. The initial experiments went through several iterations after diagnosing unstable training and weak generalization.

The final training recipe used:

- **Static hard-negative mining:** for each training query, rank the full resume corpus with the zero-shot bi-encoder and sample high-ranked non-relevant resumes as negatives.
- **Triplet loss with a 0.2 margin:** a smaller margin produced more stable optimization than the initial setting.
- **Low learning rate (`1e-6`)** to avoid aggressively changing the pretrained embedding space.
- **Gradient clipping** to stabilize updates.
- **Near-duplicate filtering** to reduce the risk of sampling false negatives.

### Fixing Evaluation Leakage

An earlier version evaluated the held-out query periodically during training and restored the checkpoint with the highest held-out NDCG@10.

Although this produced strong results, it introduced **test-set leakage**: the held-out query was effectively being used as validation data for checkpoint selection.

To obtain a cleaner estimate of generalization, checkpoint selection was replaced with a **fixed-step 10-fold Leave-One-Query-Out (LOQO) protocol**:

1. Hold out one of the 10 queries as the test query.
2. Train on the remaining 9 queries.
3. Initialize each fold independently from the original `all-mpnet-base-v2` checkpoint.
4. Train for a fixed **40 steps**, with no evaluation on the held-out query during training.
5. Evaluate the held-out query exactly once after training.
6. Repeat for all 10 queries and aggregate the metrics.

This ensures that the test query does not influence training duration or checkpoint selection.

Run it with:

```bash
python scripts/train_bi_encoder_loqo.py             # hard-negative fine-tuning, all 10 folds
python scripts/train_bi_encoder_loqo_random_neg.py  # random-negative counterpart, same protocol
python scripts/compare_loqo_results.py              # side-by-side comparison against BM25 / zero-shot
```

Each sweep saves per-fold results incrementally to `results/loqo_full_sweep.json` / `results/loqo_full_sweep_random_neg.json` (so an interrupted run doesn't lose completed folds) and fine-tuned checkpoints to `results/checkpoints/`.

## Negative Sampling Ablation

To measure the effect of negative-sampling strategy, the same MPNet bi-encoder was compared under three conditions:

| Model | Recall@10 | Recall@50 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|
| Zero-shot MPNet | 0.2893 | 0.7966 | 1.0000 | 0.9640 |
| Random-negative fine-tuning | 0.2955 | 0.8037 | 1.0000 | 0.9768 |
| **Hard-negative fine-tuning** | **0.2986** | **0.8112** | **1.0000** | **0.9873** |

All fine-tuned models use the same fixed 40-step LOQO evaluation protocol.



## Retrieval Results

All models were evaluated on the same 10 TalentCLEF queries. Fine-tuned
models use fixed 40-step Leave-One-Query-Out (LOQO) training to prevent
the held-out query from influencing checkpoint selection.

| Model | Recall@10 | Recall@50 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.2763 | 0.7970 | 0.9200 | 0.8941 |
| Zero-shot MPNet | 0.2893 | 0.7966 | 1.0000 | 0.9640 |
| Random-negative MPNet | 0.2955 | 0.8037 | 1.0000 | 0.9768 |
| **Hard-negative MPNet** | **0.2986** | **0.8112** | **1.0000** | **0.9873** |

Random negatives provided little useful supervision for this query, whereas hard negatives substantially improved top-ranked retrieval quality.

MRR remained at 1.0 across all three conditions, indicating a ceiling effect: the first relevant resume was already ranked first for nearly every query. Recall and NDCG therefore provide more informative comparisons for this benchmark.

### Takeaway

Dense retrieval substantially improved ranking quality over BM25, while fine-tuning provided additional gains. Hard-negative fine-tuning achieved the strongest overall results, reaching 0.9873 mean NDCG@10 compared with 0.8941 for BM25, 0.9640 for zero-shot MPNet, and 0.9768 for random-negative fine-tuning. The advantage of hard negatives was most apparent on difficult queries, suggesting that training against plausible but non-relevant candidates provides more informative ranking supervision than uniformly sampled negatives.

Because TalentCLEF contains only 10 labeled queries and several queries already have near-perfect zero-shot NDCG@10, these results should be interpreted as a **small-data ablation study rather than definitive evidence of generalization**. A larger query-level dataset is needed for more robust train/validation/test evaluation.
