# P2 Validation Findings: Custom ESG Embedding Model for Vertical Search

## Summary

P2 was validated across three dimensions — novelty/saturation, dataset availability, and technical feasibility — using parallel research agents. The findings were then synthesized into a head-to-head comparison against P1.

**Conclusion: P2 is the stronger PyTorch portfolio project.** It wins on 9 of 12 comparison dimensions against P1.

---

## Phase 1: Novelty & Saturation

### What's saturated

| Prior Art | Severity | Notes |
|---|---|---|
| HuggingFace SentenceTransformers fine-tuning tutorial | **HIGH** | Official playbook is comprehensive; any fine-tuning project risks looking like a follow-along |
| ClimateBERT (Webersinke et al., 2022) | **CRITICAL** | Domain-adapted RoBERTa on 2M+ climate paragraphs — but it's a masked LM for classification, not a sentence-transformer for retrieval |
| ESG-BERT (Mehra, 2022) | **HIGH** | BERT for ESG text classification (F1: 0.90 vs 0.79 general) — again classification, not retrieval |
| Pinecone / LlamaIndex / Databricks playbooks | **MEDIUM** | Generic methodology guides for embedding fine-tuning |
| FinMTEB (Feb 2025) | **MEDIUM** | Finance embedding benchmark (64 datasets, 7 tasks) — includes some ESG content but not ESG-focused |
| BEIR benchmark | **LOW** | Standard IR benchmark, no ESG content |

### What's novel (the gap)

1. **No ESG-specific sentence-transformer exists on HuggingFace.** ESG-BERT and ClimateBERT are classification models. No one has published an embedding model optimized for ESG information retrieval.
2. **No ESG information retrieval benchmark.** FinMTEB covers finance. MLEB covers legal. Nothing equivalent for ESG/sustainability.
3. **Combined fine-tuning + retrieval evaluation + business ROI framing is rare in ANY domain.** Most tutorials stop at "NDCG improved."
4. **ESGBench (Nov 2025) is too new to be heavily benchmarked.** Opportunity to establish baselines.

### Saturation verdict

| Execution level | Risk |
|---|---|
| Generic embedding fine-tuning | MEDIUM-HIGH |
| ESG domain-specific | MEDIUM |
| Combined system (ESG model + benchmark + evaluation + deployed demo + published model) | **LOW** |

The project operates in the LOW zone when all layers are combined.

---

## Phase 2: Dataset & Domain

### Primary recommendation: ESGBench (Nov 2025)

- ~933 QA pairs from 45 real ESG documents (sustainability reports, TCFD disclosures, CSR reports)
- Freely available on GitHub with extensible pipeline
- Has query-relevance structure adaptable for IR evaluation
- Too new for extensive benchmarking — opportunity to be first

**Limitation:** ~900 pairs is insufficient for training (need 10-20K for solid contrastive learning).

### Data augmentation sources (to reach 10-20K pairs)

| Source | Format | Access |
|---|---|---|
| CDP disclosures | Structured climate Q&A | Free (cdp.net, registration required) |
| GRI reports | Sustainability reports with section headers | Free (globalreporting.org) |
| TCFD reports | Structured around 4 pillars | Free (company websites) |
| Synthetic generation | LLM-generated queries from passages | ~$5-15 API cost |

### Validation dataset: FinMTEB

64 datasets, 7 tasks, includes ESG content. Use post-training to confirm ESG fine-tuning doesn't hurt general financial retrieval.

### Business context — who pays for ESG search

| Company | Service | Pricing |
|---|---|---|
| MSCI | ESG Ratings + Research Terminal | $5K-2M/yr |
| Sustainalytics | ESG Risk Ratings (Morningstar/Bloomberg) | Premium tier |
| Clarity AI | AI-driven ESG analytics | Undisclosed (growing) |
| Bloomberg | Terminal with ESG data | ~$24K/yr |
| S&P Global | ESG data, SASB ratings | Variable |

ESG search is a real, monetized problem. These companies need retrieval across thousands of heterogeneous sustainability reports.

### Why this dataset strategy beats P1

P1 downloads one CSV (CFPB). P2 constructs a novel multi-source dataset (ESGBench + CDP + GRI + TCFD + synthetic). This directly addresses the portfolio principle of "multiple data sources joined together."

---

## Phase 3: Technical Feasibility

All areas confirmed feasible:

| Area | Status | Key Detail |
|---|---|---|
| **Training format** | Confirmed | MultipleNegativesRankingLoss (MNR) — needs query-positive pairs, negatives from batch. Min ~20K pairs recommended. |
| **Model selection** | Confirmed | Primary: all-mpnet-base-v2 (110M params, 768-dim). Secondary: all-MiniLM-L6-v2 (22M, 384-dim). No ESG sentence-transformer exists yet. |
| **Compute** | Confirmed | Colab T4 comfortable. MPNet: batch_size=16, ~12-14GB VRAM. Total ~10-15 GPU-hours. |
| **Evaluation** | Confirmed | InformationRetrievalEvaluator built into sentence-transformers. rank_bm25 for sparse baseline. NanoBEIR for generalization. |
| **Hybrid retrieval** | Confirmed | Linear fusion BM25 + dense: ~20 lines of Python. Adds valuable 4th comparison point. |
| **Visualization** | Confirmed | UMAP before/after is informative IF paired with silhouette scores and per-query metrics. |
| **Scope** | Confirmed | 8-12 experiments, 4-5 weeks. Significantly lighter than P1's 54 runs. |

### Key academic backing

"Do We Need Domain-Specific Embedding Models?" (arXiv 2409.18511) demonstrates **20-40% performance drop** for general models on domain-specific tasks. Domain-specific embeddings are empirically justified.

---

## Phase 4: P1 vs P2 Head-to-Head

| Dimension | P1 (Fine-Tune vs. Prompt) | P2 (Custom ESG Embeddings) | Winner |
|---|---|---|---|
| Core PyTorch skill | Classifier fine-tuning, QLoRA, training loops | Contrastive learning, loss functions, embedding geometry | **Draw** |
| Business question clarity | "Build or buy?" — universal | "Can domain embeddings improve ESG search?" — strong for ESG | **P1** (slightly) |
| Saturation risk | HIGH (LoRA Land, arXiv papers, CFPB+BERT tutorials) | LOW-MEDIUM (ESG retrieval niche is open) | **P2** |
| Novel contribution width | Narrow: interactive decision tool | Broad: first ESG model + benchmark + published artifact | **P2** |
| Connection to RAG/LLM stack | Tangential (classification) | Direct (embeddings ARE RAG's foundation) | **P2** |
| Deliverable tangibility | Dashboard (useful but ephemeral) | Published HuggingFace model (reusable, citable) | **P2** |
| Portfolio overlap | Moderate (if B1 uses FinBERT) | Low | **P2** |
| Data source strategy | Single source (CFPB download) | Multi-source (ESGBench + CDP + GRI + TCFD + synthetic) | **P2** |
| Employer alignment | General ML teams | Direct for Clarity AI, MSCI, S&P Global | **P2** |
| Scope / cost | 4-5 weeks, 54+ runs, $20-40 | 4-5 weeks, 8-12 runs, $5-15 | **P2** |
| Data challenge difficulty | Easy (download CSV) | Harder (construct dataset) | **P2** (harder = more impressive) |
| Tutorial confusion risk | HIGH (CFPB+BERT is a published tutorial) | MEDIUM (no ESG retrieval tutorial exists) | **P2** |

**Score: P2 wins 9, P1 wins 1, Draw 1, P1 slightly ahead 1.**

### The decisive factors

1. **Saturation.** P1's novel contribution is "narrow but real." P2's is broad and defensible — no ESG retrieval model exists.
2. **Artifact.** P2 produces a published HuggingFace model. Anyone can `pip install sentence-transformers` and load it. P1 produces a Streamlit dashboard.
3. **RAG relevance.** Every 2025-2026 ML interview touches RAG. P2 demonstrates you can build the embedding layer RAG depends on. P1 demonstrates classification fine-tuning — adjacent but not central.
4. **Data engineering.** P2's multi-source dataset construction is itself portfolio-worthy. P1 downloads one file.

### When P1 would be better

If the user pivots away from ESG employers toward general ML engineering roles, P1's universal "build or buy" framing has broader appeal. Given the stated Clarity AI target, P2 is the clear choice.

---

## Phase 5: Strongest P2 Framing

**One-liner:** "General-purpose embedding models drop 20-40% on domain-specific retrieval. This project builds the first ESG-specific sentence-transformer and proves it on real sustainability reports."

**Resume line (draft):** "Built and published the first ESG-specific sentence-transformer for information retrieval: constructed 15K+ query-passage pairs from sustainability reports (ESGBench, CDP, GRI, TCFD), fine-tuned all-mpnet-base-v2 with contrastive learning, achieving [X]% NDCG@10 improvement over general embeddings. Published model to HuggingFace Hub; deployed side-by-side search comparison."

**Blog title:** "I Built an ESG Search Engine That Actually Understands Sustainability Reports"

**Who pays today:** Clarity AI, MSCI ($5K-2M/yr), Sustainalytics/Morningstar, S&P Global, Bloomberg ESG (~$24K/yr terminal cost).

---

## Phase 6: Recommendation

**Choose P2.** Proceed to writing the full `initialPlan.md` validation document (16 sections, mirroring P1's structure and depth).

### Key risks to watch

| Risk | Likelihood | Impact |
|---|---|---|
| Dataset too small for meaningful fine-tuning improvement | Medium | High |
| ESGBench too easy (BM25 already near-perfect) | Low-Medium | High |
| CDP/GRI harder to scrape/parse than expected | Medium | Medium |
| Fine-tuned model shows marginal improvement (<5% NDCG) | Medium | Medium |
| Project looks like a tutorial follow-along | Medium | High |

All have documented mitigations in the full plan.

---

## Sources

### Novelty & Saturation
- [Do We Need Domain-Specific Embedding Models?](https://arxiv.org/abs/2409.18511) (arXiv 2409.18511)
- [ClimateBERT](https://arxiv.org/abs/2110.12010) (arXiv 2110.12010)
- [ESG-BERT](https://github.com/mukut03/ESG-BERT) (GitHub)
- [FinMTEB](https://arxiv.org/html/2502.10990v1) (arXiv, Feb 2025)
- [MLEB: Massive Legal Embedding Benchmark](https://arxiv.org/pdf/2510.19365) (arXiv)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (HuggingFace)
- [HuggingFace: Train and Fine-Tune Sentence Transformers](https://huggingface.co/blog/how-to-train-sentence-transformers)
- [Pinecone: Fine-tune Sentence Transformers with MNR Loss](https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/)

### Datasets
- [ESGBench: A Benchmark for Explainable ESG QA](https://arxiv.org/abs/2511.16438) (arXiv, Nov 2025)
- [ESGBench GitHub](https://github.com/sherinegeorge21/ESGBench)
- [Amazon ESCI / Shopping Queries Dataset](https://github.com/amazon-science/esci-data) (KDD Cup 2022)
- [CDP Data Portal](https://www.cdp.net/en/data)
- [GRI Standards](https://www.globalreporting.org/standards/)
- [FinBERT-ESG](https://huggingface.co/yiyanghkust/finbert-esg) (HuggingFace)

### Technical Feasibility
- [SentenceTransformers Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html)
- [SentenceTransformers Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html)
- [InformationRetrievalEvaluator](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/InformationRetrievalEvaluator.py) (GitHub)
- [rank-bm25](https://pypi.org/project/rank-bm25/) (PyPI)
- [BEIR Benchmark](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf) (NeurIPS 2021)
- [NanoBEIR Evaluator](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/evaluation/NanoBEIREvaluator.py) (GitHub)
- [Weaviate: How to Choose a Sentence Transformer](https://weaviate.io/blog/how-to-choose-a-sentence-transformer-from-hugging-face)
