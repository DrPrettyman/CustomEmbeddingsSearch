# P2: Custom ESG Embedding Model for Vertical Search

## Full Project Plan

---

## 1. The Business Problem

An ESG analyst at an asset manager, a sustainability consultant, or a compliance officer at a large corporation wants to answer questions like:

- What did Company X disclose about its Scope 3 emissions methodology last year?
- Which companies in our portfolio have set science-based targets aligned with a 1.5°C pathway?
- What water stress risks has Company Y identified in its operations, and what mitigation measures are in place?
- How does Company Z's board governance of climate risk compare to its sector peers?

Today, finding these answers requires reading through hundreds of sustainability reports — PDF-heavy, inconsistently structured documents ranging from 50 to 300 pages — or paying for commercial platforms:

- **MSCI ESG Research:** $5K–2M/year depending on coverage and features
- **Sustainalytics (Morningstar):** Premium-tier institutional pricing
- **Clarity AI:** AI-driven ESG analytics, undisclosed pricing (fast-growing)
- **Bloomberg Terminal ESG module:** ~$24K/year (part of terminal subscription)
- **S&P Global ESG data:** Variable institutional pricing

These platforms all face the same underlying technical challenge: retrieving specific information from domain-specific text where general-purpose search fails. Sustainability reports use specialised vocabulary — "Scope 3 Category 15," "TCFD Pillar 2," "GRI 305-1," "science-based targets," "just transition" — that general-purpose embedding models were not trained on. When a general model encodes "materiality assessment," it doesn't know this means something specific and different in ESG context versus financial auditing versus product engineering.

Published research confirms this: "Do We Need Domain-Specific Embedding Models?" (arXiv 2409.18511) demonstrates a **20–40% performance drop** when general-purpose models are applied to domain-specific retrieval tasks. The embedding layer is the foundation of every RAG system, every semantic search engine, and every retrieval-based QA pipeline. If the embeddings don't understand the domain, everything downstream suffers.

**Estimated business impact (with assumptions stated):** An ESG analytics platform serving 50 institutional clients might process ~500 analyst queries per day across its user base. If general-purpose embeddings return a relevant result in the top 3 for 60% of queries (consistent with the 20–40% domain performance gap), and a domain-specific model raises that to 75%, that's 75 additional queries per day where the analyst finds what they need without manual report reading. At an estimated 15 minutes saved per successful retrieval and analyst billing rates of €80–150/hour, that's roughly €1,400–2,800/day in recovered productivity — or **€350K–700K/year** across the platform. Even at conservative assumptions (smaller client base, lower query volume, smaller hit-rate improvement), the value of closing the embedding quality gap on ESG text is significant. These platforms already charge $5K–2M/year for access precisely because manual search through sustainability reports is expensive.

**Resume line (draft):** "Built and published the first ESG-specific sentence-transformer for information retrieval: constructed 15K+ query-passage pairs from sustainability reports (ESGBench, CDP, GRI, TCFD), fine-tuned all-mpnet-base-v2 with contrastive learning, achieving [X]% NDCG@10 improvement over general embeddings. Published model to HuggingFace Hub; deployed side-by-side search comparison on real sustainability reports."

---

## 2. Why This Problem, For This Portfolio

This project is selected for specific strategic reasons:

**Domain alignment with target roles.** Clarity AI builds AI-driven ESG analytics. MSCI, S&P Global, and Sustainalytics all need retrieval systems that understand sustainability text. Any role involving ESG data, sustainability reporting, or responsible investment analytics benefits from demonstrated ability to build domain-specific retrieval. This project signals: "I can build the search infrastructure that ESG platforms depend on."

**Fills the PyTorch / deep learning gap.** The existing portfolio has classical ML (Steam recommendation engine), data engineering (Private Label opportunity scoring), and RAG/LLM (Food Safety regulatory intelligence). None involve training a neural network from scratch or fine-tuning with PyTorch. P2 demonstrates contrastive learning, loss function design, GPU training loops, embedding space geometry, and model publishing — the core deep learning engineering skills job descriptions ask for.

**Produces a published, reusable artifact.** Unlike a dashboard (ephemeral) or an analysis notebook (static), P2 publishes a model to HuggingFace Hub. Anyone can `pip install sentence-transformers` and load it. This is citable, reusable, and visible to the community — a fundamentally different kind of portfolio output that signals open-source contribution ability.

**Direct connection to the RAG/LLM stack.** Embeddings are the foundation of every RAG system. The Food Safety project (L1) uses a vector store for retrieval — P2 demonstrates the ability to *build the embedding model* that powers retrieval. Together, L1 and P2 show: "I can build RAG systems AND I can improve the underlying embedding layer when general models aren't good enough."

**Multi-source data engineering.** P2 doesn't download a single CSV. It constructs a novel training dataset from ESGBench, CDP disclosures, GRI reports, TCFD frameworks, and synthetic LLM-generated queries. This data construction pipeline is itself portfolio-worthy — directly addressing the portfolio principle of "multiple data sources joined together."

**Ties to the ESG/sustainability domain.** The Private Label project analyses food products including Nutri-Score (a sustainability-adjacent metric). L1 covers EU food safety regulation (which increasingly incorporates sustainability requirements). P2 extends the portfolio's European/sustainability thread into the machine learning layer.

---

## 3. Novelty Assessment

### What already exists

| Project / Model | What it does | How P2 differs |
|---|---|---|
| **ESG-BERT** (Mehra, 2022) | BERT fine-tuned for ESG text **classification** (26 ESG categories, F1: 0.90 vs. 0.79 general BERT). GitHub: ~200 stars. | Classification model, not a retrieval/embedding model. Cannot be used for semantic search. Different task entirely. |
| **ClimateBERT** (Webersinke et al., 2022) | Domain-adapted RoBERTa on 2M+ climate paragraphs. Masked LM for downstream **classification** tasks (climate sentiment, TCFD classification). | Masked language model, not a sentence-transformer. Produces token embeddings, not sentence embeddings optimised for similarity search. |
| **FinBERT-ESG** (HuggingFace: yiyanghkust/finbert-esg) | FinBERT variant for ESG **sentiment analysis** (positive/negative/neutral). | Sentiment classifier. Does not produce embeddings for retrieval. |
| **FinMTEB** (Feb 2025) | Financial embedding benchmark: 64 datasets, 7 tasks. Includes some ESG-adjacent content. | A benchmark, not a model. Covers finance broadly — no ESG-specific retrieval task. P2 could contribute an ESG subset to this ecosystem. |
| **Generic sentence-transformer fine-tuning tutorials** (HuggingFace, Pinecone, Databricks) | Step-by-step guides for fine-tuning on custom domains. | Tutorials, not ESG-specific implementations. Following one produces a tutorial clone. P2's domain, dataset construction, and evaluation framework go far beyond any tutorial. |
| **ESGBench** (Nov 2025) | QA benchmark for explainable ESG analysis: ~933 QA pairs from 45 real sustainability reports. | A benchmark dataset, not a model. Too new for extensive benchmarking — opportunity to establish retrieval baselines. |

### What does NOT exist (confirmed via HuggingFace, GitHub, arXiv, Kaggle, Medium search)

- **Zero** ESG-specific sentence-transformers published on HuggingFace
- **Zero** ESG information retrieval benchmarks (FinMTEB covers finance broadly; MLEB covers legal; nothing for ESG/sustainability)
- **Zero** published evaluations of embedding retrieval quality on sustainability reports
- **Zero** portfolio projects combining ESG embedding fine-tuning + retrieval evaluation + published model

### Why the gap persists

ESG-BERT and ClimateBERT solved the classification problem — "is this text about climate/ESG?" — because classification was the first NLP task the ESG analytics industry needed. But the industry has since moved to retrieval-based systems (RAG, semantic search over document corpora), and nobody has built the retrieval-optimised embedding layer for ESG text. The classification models can't be used for retrieval because they produce token-level embeddings, not sentence-level similarity embeddings. Sentence-transformers require contrastive training on (query, relevant passage) pairs — and no such ESG training dataset existed until ESGBench (Nov 2025) provided the seed.

### Our differentiation (4 layers)

1. **First ESG retrieval embedding model.** Not classification (ESG-BERT), not masked LM (ClimateBERT), not sentiment (FinBERT-ESG) — a sentence-transformer optimised for ESG information retrieval.
2. **Novel training dataset.** Constructed from 4+ sources (ESGBench + CDP + GRI + TCFD + synthetic). This specific combination doesn't exist as a published dataset.
3. **Rigorous retrieval evaluation.** NDCG@10, MRR, Recall@K against BM25, off-the-shelf dense, and hybrid baselines. Per-query failure analysis. Cross-domain generalisation check against FinMTEB.
4. **Published artifact.** Model on HuggingFace Hub — not a notebook, not a dashboard, but a reusable model anyone can load.

---

## 4. Data Sources

### Primary Training Data: Constructed ESG Query-Passage Dataset

The core challenge: contrastive learning with MultipleNegativesRankingLoss needs (query, relevant_passage) pairs — ideally 10–20K for solid fine-tuning. No single ESG dataset provides this at scale. The training dataset is constructed from multiple sources:

#### Source 1: ESGBench (Nov 2025)

**Access:** GitHub (sherinegeorge21/ESGBench), freely available.

**Content:** ~933 QA pairs derived from 45 real ESG documents (sustainability reports, TCFD disclosures, CSR reports). Each pair has a question and an answer traceable to specific passages in the source documents.

**What we extract:** Convert QA pairs into (query, relevant_passage) format. The question becomes the query; the source passage (not the answer) becomes the relevant document. This gives ~900 training pairs directly.

**Why it matters:** Real questions from real ESG documents. Not synthetic, not templated. This is the highest-quality seed data.

#### Source 2: CDP Climate Disclosures

**Access:** CDP Data Portal (cdp.net), free registration required. Companies respond to structured questionnaires about climate risks, emissions, targets, and governance.

**Content:** Structured Q&A format — companies answer specific questions (e.g., "C2.1a: How does your organization define substantive financial or strategic impact on your business?"). Thousands of company responses available.

**What we extract:** CDP questions become queries. Company responses become relevant passages. The structured Q&A format maps naturally to (query, passage) pairs. Target: 3,000–5,000 pairs from CDP disclosures.

**Why it matters:** CDP data is structured, domain-specific, and covers climate/environmental ESG topics in depth. The Q&A format requires minimal reformatting.

#### Source 3: GRI Sustainability Reports

**Access:** GRI (globalreporting.org) provides the reporting standards freely. Actual reports are published on company websites and the GRI database.

**Content:** Sustainability reports structured around GRI disclosure topics (e.g., GRI 305: Emissions, GRI 403: Occupational Health and Safety). Reports follow standardised section headers.

**What we extract:** GRI disclosure topic descriptions become queries (e.g., "What are the organisation's direct greenhouse gas emissions?"). Corresponding report sections become relevant passages. Target: 2,000–4,000 pairs from GRI-structured reports.

**Why it matters:** GRI is the most widely used sustainability reporting framework globally. Reports structured around GRI topics provide natural query-passage alignment.

#### Source 4: TCFD-Aligned Reports

**Access:** Company websites. TCFD reports follow a structured framework: Governance, Strategy, Risk Management, Metrics & Targets.

**Content:** Reports organised around TCFD's four pillars and eleven recommended disclosures. Highly structured.

**What we extract:** TCFD recommended disclosure questions become queries (e.g., "How does the organisation identify and assess climate-related risks?"). Report sections addressing each recommendation become passages. Target: 1,000–2,000 pairs.

**Why it matters:** TCFD is now mandatory in many jurisdictions (UK, Japan, EU via CSRD). The structured framework provides high-quality query-passage alignment.

#### Source 5: Synthetic Query Generation (Augmentation)

**Access:** LLM API calls (~$5–15 cost).

**Method:** For passages from Sources 2–4 that don't have natural query counterparts, generate synthetic queries using an LLM:
- Input: ESG passage from a sustainability report
- Prompt: "Generate 3 diverse search queries that an ESG analyst would use to find this passage. Vary query specificity: one broad, one medium, one narrow."
- Output: 3 (query, passage) pairs per passage

**Target:** 3,000–5,000 synthetic pairs to fill gaps and increase diversity.

**Why it matters:** Synthetic augmentation is standard practice in embedding fine-tuning (documented in SentenceTransformers training guides). The LLM-generated queries simulate real analyst search behaviour, adding query diversity that structured Q&A sources lack.

### Training Data Summary

| Source | Pairs (estimated) | Quality | Format effort |
|---|---|---|---|
| ESGBench | ~900 | High (real QA) | Low (already structured) |
| CDP disclosures | 3,000–5,000 | High (structured Q&A) | Medium (parsing required) |
| GRI reports | 2,000–4,000 | Medium-High | Medium (section extraction) |
| TCFD reports | 1,000–2,000 | High (structured) | Medium |
| Synthetic | 3,000–5,000 | Medium (LLM-generated) | Low (API calls) |
| **Total** | **10,000–17,000** | | |

This is within the recommended range for contrastive learning with MNR loss (10–20K pairs).

### Evaluation Data: Held-Out Test Set

**Construction:** Reserve 15–20% of the highest-quality pairs (ESGBench + CDP) as the evaluation set. These are never seen during training.

**Structure:** For each query in the evaluation set, the relevant passage is the positive. All other passages in the corpus serve as the retrieval candidates. This creates a realistic retrieval task: "given this ESG query, find the right passage among thousands."

**Target:** ~150–200 evaluation queries with known relevant passages.

### Generalisation Check: FinMTEB

**Purpose:** Confirm that ESG fine-tuning doesn't catastrophically degrade performance on general financial text.

**Method:** Run the fine-tuned model on a subset of FinMTEB tasks (particularly those adjacent to ESG) and compare against the base model's performance. Acceptable outcome: minimal degradation (<5% drop) on non-ESG financial tasks.

---

## 5. What the System Builds

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│            1. Dataset Construction Pipeline                  │
│                                                             │
│  ESGBench → parse QA pairs                                  │
│  CDP disclosures → parse structured Q&A                     │
│  GRI reports → extract section headers + content            │
│  TCFD reports → extract disclosure sections                 │
│  → Merge + deduplicate + quality filter                     │
│  → Split: train (80-85%) / eval (15-20%)                    │
│  → LLM synthetic augmentation on train set                  │
│                                                             │
│  Output: data/processed/train_pairs.jsonl                   │
│          data/processed/eval_queries.jsonl                  │
│          data/processed/eval_corpus.jsonl                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            2. Baseline Evaluation (Before Training)          │
│                                                             │
│  Run evaluation queries against eval corpus using:          │
│    A. BM25 (sparse baseline) — rank_bm25                    │
│    B. all-mpnet-base-v2 (dense baseline, off-the-shelf)     │
│    C. all-MiniLM-L6-v2 (smaller dense baseline)             │
│    D. Hybrid: BM25 + dense linear fusion                    │
│                                                             │
│  Metrics per method:                                        │
│    NDCG@10, MRR@10, Recall@10, Recall@100                  │
│    Per-query results (for later failure analysis)           │
│                                                             │
│  This is the "before" in the before/after comparison.       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            3. Fine-Tuning Pipeline (PyTorch)                 │
│                                                             │
│  Model: all-mpnet-base-v2 (110M params, 768-dim)            │
│  Loss: MultipleNegativesRankingLoss (MNR)                   │
│  Hardware: Google Colab T4 (batch_size=16, ~12-14GB VRAM)   │
│                                                             │
│  Training experiments:                                      │
│    Exp 1: MNR loss, default hyperparameters                 │
│    Exp 2: MNR loss + hard negative mining                   │
│    Exp 3: MNR loss + hard negatives + learning rate sweep   │
│    Exp 4: Different training data compositions              │
│           (ESGBench-only, CDP-only, all combined)           │
│    Exp 5: Training data size ablation                       │
│           (2K, 5K, 10K, 15K pairs)                          │
│    Exp 6: Secondary model (all-MiniLM-L6-v2) for size       │
│           comparison                                        │
│                                                             │
│  Total: 8-12 experiments, ~10-15 GPU-hours                  │
│                                                             │
│  Each experiment evaluated on held-out eval set.            │
│  Best model selected by NDCG@10 on eval set.               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            4. Post-Training Evaluation Suite                  │
│                                                             │
│  A. ESG retrieval (primary metric):                         │
│     Fine-tuned vs. baselines on held-out eval queries.      │
│     NDCG@10, MRR@10, Recall@10, Recall@100                 │
│                                                             │
│  B. Hybrid retrieval:                                       │
│     BM25 + fine-tuned dense (linear fusion)                 │
│     vs. BM25 + off-the-shelf dense                          │
│     → Does fine-tuning help even in hybrid mode?            │
│                                                             │
│  C. Per-query failure analysis:                             │
│     Which queries improved? Which degraded? Why?            │
│     Categorise by ESG topic, query type, passage length.    │
│                                                             │
│  D. Embedding space visualisation:                          │
│     UMAP before vs. after fine-tuning.                      │
│     Cluster coherence: silhouette scores by ESG topic.      │
│                                                             │
│  E. FinMTEB generalisation check:                           │
│     Run fine-tuned model on FinMTEB subset.                 │
│     Verify ESG specialisation doesn't break general         │
│     financial retrieval.                                    │
│                                                             │
│  F. Training data ablation analysis:                        │
│     Plot NDCG@10 vs. training set size.                     │
│     "How much ESG data do you need before fine-tuning       │
│     beats general embeddings?"                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            5. Output: Published Model + Search Demo          │
│                                                             │
│  HuggingFace Hub:                                           │
│    - Published model with model card                        │
│    - Training details, evaluation results, limitations      │
│    - Usage example in model card README                     │
│                                                             │
│  Streamlit app:                                             │
│    - Type an ESG query                                      │
│    - See ranked results from 4 methods side-by-side:        │
│      BM25 | Off-the-shelf | Fine-tuned | Hybrid            │
│    - Relevance scores + source document citations           │
│    - Embedding space visualisation                          │
│                                                             │
│  FastAPI:                                                   │
│    - POST /search with query + method selection             │
│    - GET /compare for side-by-side evaluation               │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture, Not the "RAG Chatbot" Alternative

The naive approach — "build a chatbot over ESG reports using LangChain" — demonstrates only that you can call APIs. This project rejects the chatbot pattern deliberately:

| | ESG RAG Chatbot | Custom ESG Embedding Model |
|---|---|---|
| **What it proves** | "I can build a RAG pipeline" | "I can improve the retrieval layer RAG depends on" |
| **Artifact** | A running app (ephemeral) | A published model on HuggingFace (permanent, reusable) |
| **Evaluation** | Subjective ("the answers look good") | Quantitative (NDCG@10, MRR, Recall@K against baselines) |
| **Novelty** | Saturated (thousands of RAG chatbot repos) | Novel (zero ESG sentence-transformers exist) |
| **Depth** | Uses embeddings as a black box | Understands and improves the embedding layer itself |
| **Interview story** | "I built a chatbot" | "I measured a 20-40% retrieval gap in ESG text, then closed it" |

The fine-tuned embedding model is the infrastructure that makes ESG RAG systems work. Building the infrastructure is harder and more impressive than building the application on top of it.

### PyTorch Components (Used Naturally)

| Component | Purpose | Why it's the right tool |
|---|---|---|
| **SentenceTransformers (PyTorch)** | Model loading, fine-tuning, evaluation | The standard library for sentence embedding fine-tuning. PyTorch backend gives full control over training. |
| **MultipleNegativesRankingLoss** | Contrastive loss function | Standard for embedding fine-tuning when you have (query, positive) pairs. In-batch negatives are efficient and effective. |
| **Hard negative mining** | Improve training signal quality | Easy negatives (random passages) don't teach the model much. Hard negatives (passages that are similar but wrong) force the model to learn domain-specific distinctions. |
| **Custom DataLoader** | Handle multi-source training data | Different sources (ESGBench, CDP, GRI) have different formats. Custom loader handles heterogeneous data. |
| **InformationRetrievalEvaluator** | Standard IR metrics during training | Built into SentenceTransformers. Computes NDCG, MRR, Recall during training for model selection. |
| **UMAP + matplotlib** | Embedding space visualisation | Before/after visualisation is the most compelling evidence of what fine-tuning actually does to the embedding space. |
| **HuggingFace Hub API** | Model publishing | Push trained model to Hub with model card, making it discoverable and loadable. |
| **ONNX Runtime** | Production inference optimisation | Export fine-tuned model to ONNX for CPU-optimised inference. Demonstrates production deployment thinking beyond "it works on my GPU." |

### What This Is NOT

- Not a RAG chatbot. No LLM, no generation, no conversation. Input: search query. Output: ranked relevant passages with scores.
- Not a classification model. ESG-BERT classifies text into ESG categories. P2 embeds text for similarity-based retrieval. Different task, different architecture, different training objective.
- Not just "I fine-tuned a sentence-transformer." The contribution is the combination: ESG domain focus + novel multi-source dataset + rigorous retrieval evaluation + published model + business framing. Any one of these alone would be a tutorial follow-along. Together, they're a project.

---

## 6. Technical Challenges Worth Documenting

These are the hard problems that make this project portfolio-worthy:

### 6a. Training Dataset Construction from Heterogeneous Sources

The most labour-intensive and arguably most impressive part of the project. Each source has different structure:

- **ESGBench:** JSON with question, answer, source document, and evidence spans. Needs reformatting to (query, passage) pairs where the passage is the source text, not the extracted answer.
- **CDP:** Structured questionnaire responses. Questions are standardised but responses vary in length, quality, and language. Need to filter for English, remove boilerplate, and extract substantive Q&A pairs.
- **GRI:** Reports vary enormously in structure. Some follow GRI Content Index closely; others embed GRI disclosures across narrative sections. Extraction requires matching GRI topic numbers to report sections.
- **TCFD:** Relatively standardised (4 pillars, 11 recommendations), but implementation varies. Need to align report sections to TCFD disclosure questions.

**Approach:** Build a source-specific parser for each, outputting a common schema: `{query: str, passage: str, source: str, topic: str}`. Quality filter: remove pairs where passages are <50 words (too short for meaningful retrieval) or >2,000 words (too long — should be sub-chunked). Deduplicate on passage text (some reports reuse text across frameworks).

**Blog-worthy insight:** "Building an ESG retrieval dataset from scratch: what I learned parsing CDP disclosures, GRI reports, and TCFD frameworks into training data."

### 6b. Hard Negative Mining Strategy

The most impactful training decision after dataset size. With MultipleNegativesRankingLoss, the model learns from in-batch negatives — random passages that happen to be in the same batch. These are "easy" negatives: a passage about water stress is obviously different from a passage about board governance.

Hard negatives are passages that are *similar* to the relevant one but *wrong* — and they force the model to learn finer distinctions:

- "Scope 1 emissions from our operations decreased 12% in 2024" vs. "Scope 3 emissions from our supply chain decreased 12% in 2024" — same structure, different scope category
- "We have set a science-based target for 2030" vs. "We are evaluating whether to set a science-based target" — target vs. aspiration
- "Our board has oversight of climate risk" vs. "Our board has been briefed on climate risk" — oversight vs. awareness

**Approach:** Two-stage training. Stage 1: MNR loss with in-batch negatives (warms up the model). Stage 2: Mine hard negatives using the Stage 1 model — for each query, retrieve the top-K passages and use the highest-ranked *wrong* passages as hard negatives. Re-train with these harder examples.

**Blog-worthy insight:** "Why hard negative mining matters for ESG embeddings: the difference between 'obviously wrong' and 'subtly wrong' passages."

### 6c. Evaluating Whether Fine-Tuning Actually Helped

The project's credibility depends on whether the fine-tuned model measurably outperforms baselines. This requires careful evaluation design:

**The BM25 ceiling problem.** If the evaluation queries contain the same keywords as the relevant passages (likely for structured Q&A sources), BM25 may perform near-perfectly — leaving no room for dense retrieval to improve. This is especially likely for CDP data where questions and answers share specific terminology.

**Mitigation:** Include evaluation queries of varying difficulty:
- **Easy (keyword-overlap):** "What are Company X's Scope 1 emissions?" — BM25 handles this fine
- **Medium (paraphrase):** "How does Company X measure its direct carbon footprint?" — same concept, different words. Dense should beat BM25 here.
- **Hard (conceptual):** "What climate transition risks has Company X identified?" — requires understanding that "transition risk" connects to regulatory changes, market shifts, technology disruption. Dense should excel.

Reporting results disaggregated by query difficulty shows *where* fine-tuning helps — much more informative than a single NDCG number.

**The marginal improvement risk.** If fine-tuning improves NDCG@10 by 2%, the project looks underwhelming. Mitigations:
- Training data ablation (Experiment 5) shows the *trajectory* — even if the current dataset is too small for dramatic improvement, the curve should show that more data would help
- Per-query analysis identifies *which query types* benefit most, providing actionable insight even if the aggregate gain is modest
- Hybrid retrieval may show larger gains (fine-tuned dense + BM25 > off-the-shelf dense + BM25) because the fine-tuned model's improvements complement BM25's strengths

### 6d. Embedding Space Geometry and Visualisation

UMAP plots of before vs. after fine-tuning are the most visually compelling evidence — but they can be misleading without quantitative backing:

- UMAP is a non-linear dimensionality reduction that can create apparent clusters even from random data
- Visual "improvement" may not correspond to retrieval improvement

**Approach:** Pair every UMAP visualisation with quantitative cluster metrics:
- Silhouette score by ESG topic (E, S, G) — does fine-tuning make topic clusters tighter?
- Average intra-cluster distance vs. inter-cluster distance — are ESG topics more separated after fine-tuning?
- Retrieval-relevant analysis: for queries that improved, show how their relevant passages moved in embedding space

**Blog-worthy insight:** "What fine-tuning actually does to ESG embedding space — visualised and measured."

### 6e. Ensuring the Model Doesn't Catastrophically Forget

Fine-tuning on ESG data risks degrading the model's ability to handle non-ESG text. A model that's great at ESG retrieval but useless for everything else has limited practical value — in production, an ESG search system may also need to search financial filings, news articles, and general business text.

**Approach:** Run the fine-tuned model on a subset of FinMTEB tasks (the financial embedding benchmark). Compare against the base model. Acceptable outcome: <5% degradation on non-ESG financial tasks. If degradation is larger, document it honestly and discuss the trade-off (specialist vs. generalist model).

**Blog-worthy insight:** "The specialisation tax: what my ESG model forgot when it learned sustainability."

---

## 7. Evaluation Framework

This is where the project separates from "I fine-tuned a sentence-transformer." The evaluation has six layers:

### Layer 0: Dataset Quality Audit

**Question:** Is the constructed training dataset clean, diverse, and representative of real ESG retrieval needs?

**Metrics:**
- Total pairs by source (ESGBench, CDP, GRI, TCFD, synthetic)
- Topic distribution across E, S, G pillars — are we balanced or skewed?
- Query length distribution (too short = trivial; too long = unrealistic)
- Passage length distribution (too short = uninformative; too long = retrieval noise)
- Duplicate rate (exact and near-duplicate passages across sources)
- Language quality (any non-English contamination?)

**Presentation:** Summary statistics table + distribution plots. Honest about any imbalances (e.g., Environmental topics likely dominate because CDP and TCFD are climate-focused).

### Layer 1: Baseline Performance

**Question:** How well do existing methods retrieve ESG information, before any fine-tuning?

**Methods evaluated:**
- **BM25** (sparse): The keyword-matching baseline. Uses rank_bm25.
- **all-mpnet-base-v2** (dense, off-the-shelf): The general-purpose sentence-transformer we'll fine-tune.
- **all-MiniLM-L6-v2** (dense, smaller): Lighter model for comparison.
- **Hybrid** (BM25 + off-the-shelf dense): Linear score fusion.

**Metrics:** NDCG@10, MRR@10, Recall@10, Recall@100.

**Presentation:** Baseline results table. This establishes the "before" that all fine-tuning experiments are measured against.

**Target:** BM25 should be competitive on keyword-heavy queries. Off-the-shelf dense should outperform BM25 on paraphrase/conceptual queries. These are expected patterns — deviations indicate dataset issues.

### Layer 2: Fine-Tuning Improvement

**Question:** Does ESG-specific fine-tuning measurably improve retrieval quality?

**Metrics (per experiment):**
- NDCG@10, MRR@10, Recall@10, Recall@100
- Δ vs. best baseline (absolute and percentage improvement)
- Statistical significance: bootstrap confidence interval on NDCG@10 difference

**Presentation:** Results table with best experiment highlighted:

| Method | NDCG@10 | MRR@10 | Recall@10 | Recall@100 | Δ NDCG vs. baseline |
|---|---|---|---|---|---|
| BM25 | [X] | [X] | [X] | [X] | — |
| mpnet (off-the-shelf) | [X] | [X] | [X] | [X] | — |
| **mpnet (fine-tuned, best)** | [X] | [X] | [X] | [X] | **+[X]%** |
| Hybrid (BM25 + fine-tuned) | [X] | [X] | [X] | [X] | +[X]% |

**Target:** NDCG@10 improvement ≥ 10% over off-the-shelf dense. If <5%, the finding is "general-purpose models are already adequate for ESG retrieval at this dataset scale" — document why (likely dataset too small or BM25-friendly query distribution).

### Layer 3: Training Data Ablation

**Question:** How much ESG training data do you need before fine-tuning beats general embeddings?

**Method:**
1. Train models at data sizes: 1K, 2K, 5K, 10K, 15K pairs
2. Evaluate each on the held-out test set
3. Plot: training set size (x-axis) vs. NDCG@10 (y-axis)
4. Mark the crossover point where fine-tuned > off-the-shelf baseline

**Presentation:** The learning curve chart — one of the most valuable deliverables. Shows:
- Minimum viable training set size for ESG embedding improvement
- Whether the curve is still rising at 15K (suggesting more data would help further)
- Diminishing returns analysis

**Why this matters:** Practical guidance for anyone considering domain-specific embedding fine-tuning. "You need at least [N]K pairs before fine-tuning outperforms general models for ESG text."

### Layer 4: Per-Query Failure Analysis

**Question:** When the fine-tuned model fails (or doesn't improve), WHY?

**Method:**
- For each evaluation query, record rank of the relevant passage for all methods
- Identify queries where fine-tuned model **improved** over baseline (success cases)
- Identify queries where fine-tuned model **degraded** from baseline (failure cases)
- Identify queries where **all methods fail** (hard cases)

**Categorise failures by:**
- **Query type:** keyword-overlap, paraphrase, conceptual, multi-hop
- **ESG topic:** Environmental, Social, Governance
- **Passage characteristics:** length, specificity, technical density
- **Source:** ESGBench, CDP, GRI, TCFD

**Presentation:** Failure mode breakdown with specific examples:
- "Fine-tuned model improves most on conceptual queries about climate risk (+[X]% NDCG), but shows no improvement on governance queries — likely because training data is skewed toward Environmental topics"
- "BM25 outperforms all dense methods on highly technical queries containing specific standard numbers (GRI 305-1, TCFD Pillar 2) — keyword matching is correct here"

This is the section hiring managers remember.

### Layer 5: Generalisation Check (FinMTEB)

**Question:** Does ESG specialisation come at the cost of general financial retrieval ability?

**Method:** Run fine-tuned model vs. base model on FinMTEB retrieval tasks (subset).

**Metrics:** NDCG@10 per FinMTEB task, fine-tuned vs. base.

**Presentation:** Table showing task-by-task comparison. Highlight any tasks where degradation > 5%.

**Target:** Mean degradation < 5% across FinMTEB tasks. If degradation is larger, this is an honest finding about the specialisation-generalisation trade-off.

### Layer 6: Inference Latency Benchmark

**Question:** Does domain fine-tuning affect inference speed? How does the model compare to baselines in a production-relevant setting?

**Method:**
- Benchmark encoding time (ms per query, ms per passage) for: base mpnet, fine-tuned mpnet, MiniLM, and ONNX-exported fine-tuned model (see Section 8)
- Measure on both CPU (single-core, simulating a budget deployment) and GPU (Colab T4, simulating production)
- Batch encoding throughput: passages/second at batch sizes 1, 16, 64, 256
- Compare against BM25 query time on the same corpus

**Metrics:** Mean encoding latency (ms), throughput (passages/sec), p95 latency.

**Presentation:** Latency comparison table. Expected finding: fine-tuned model has identical latency to base (same architecture, same parameter count). ONNX export provides [X]% speedup on CPU. BM25 is faster than all dense methods but lacks semantic understanding.

**Why this matters:** A hiring manager at an ESG platform cares about retrieval quality AND query latency. Showing that fine-tuning improves quality at zero latency cost (or demonstrating the quality-latency trade-off with ONNX) is a production-relevant finding.

---

## 8. Deployment Plan

### HuggingFace Model Publication (Primary Artifact)

The published model is the most important output — more important than the demo app.

**Model card includes:**
- Model description and intended use case
- Training data sources and construction methodology
- Training procedure (loss function, hyperparameters, hardware)
- Evaluation results (NDCG@10, MRR, comparison table)
- Limitations and biases (Environmental topic bias, English-only, temporal scope)
- Usage example:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("username/esg-retrieval-mpnet-v1")

queries = ["What are the company's Scope 3 emissions?"]
passages = ["In 2024, our Scope 3 emissions totaled...", "Board oversight of ESG..."]

query_embeddings = model.encode(queries)
passage_embeddings = model.encode(passages)

# Compute similarity
from sentence_transformers.util import cos_sim
scores = cos_sim(query_embeddings, passage_embeddings)
```

### Streamlit App (Interactive Demo)

A search interface with side-by-side comparison:

**Search panel (top):**
- Text input: type an ESG query
- Example queries provided as clickable chips (pre-populated with representative queries across E, S, G topics)

**Results panel (bottom):**
- Four columns: BM25 | Off-the-shelf | Fine-tuned | Hybrid
- Each column shows top-5 results with relevance scores
- Relevant passages highlighted if known (for evaluation queries)
- Colour-coded: green for matches across methods, amber for differences

**Sidebar:**
- Embedding space visualisation (UMAP, toggleable before/after)
- Evaluation metrics summary
- Training data composition pie chart

**Design goal:** A hiring manager opens the app, types "climate risk governance," and immediately sees that the fine-tuned model surfaces TCFD governance disclosures that the off-the-shelf model misses. Visual, immediate, no configuration.

### FastAPI Endpoints (Secondary)

```
POST /search
{
  "query": "What are the company's Scope 3 emissions?",
  "method": "fine_tuned",  // or "bm25", "off_the_shelf", "hybrid"
  "top_k": 10
}

→ 200 OK
{
  "query": "What are the company's Scope 3 emissions?",
  "method": "fine_tuned",
  "results": [
    {
      "passage": "In 2024, our total Scope 3 emissions...",
      "score": 0.89,
      "source": "Company_X_Sustainability_Report_2024",
      "esg_topic": "Environmental"
    },
    ...
  ]
}
```

```
GET /compare?query=climate+risk+governance

→ 200 OK
{
  "query": "climate risk governance",
  "methods": {
    "bm25": {"results": [...], "ndcg_10": null},
    "off_the_shelf": {"results": [...], "ndcg_10": null},
    "fine_tuned": {"results": [...], "ndcg_10": null},
    "hybrid": {"results": [...], "ndcg_10": null}
  }
}
```

### ONNX Export (Production Optimisation)

Export the best fine-tuned model to ONNX format for CPU-optimised inference:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("path/to/best-model")
model.export_onnx("onnx_model/", optimize=True)
```

**Why this matters:**
- Demonstrates production deployment thinking — real search systems run on CPU fleets, not GPUs
- ONNX Runtime typically provides 2–4x speedup on CPU vs. vanilla PyTorch
- The latency benchmark (Section 7, Layer 6) quantifies the actual speedup on this model
- Shows awareness of the inference cost side of the "is fine-tuning worth it?" question

**Deliverable:** ONNX model published alongside the PyTorch model on HuggingFace Hub, with usage example in the model card.

### Docker + Render (Production)

- `Dockerfile` using `python:3.11-slim`
- Model weights bundled in image (or downloaded from HuggingFace on startup); ONNX model used for inference in the deployed app
- Pre-computed evaluation corpus embeddings baked in (no re-encoding at demo time)
- Deploy to Render free tier for a live demo URL
- Health check endpoint at `/health`

---

## 9. Project Structure

```
esg-embedding-search/
├── README.md                          # Business-first, < 500 words
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/                           # Source documents (gitignored if large)
│   │   ├── esgbench/                 # ESGBench QA data
│   │   ├── cdp/                      # CDP disclosure responses
│   │   ├── gri/                      # GRI-structured reports
│   │   └── tcfd/                     # TCFD-aligned reports
│   ├── processed/                     # Constructed training/eval data
│   │   ├── train_pairs.jsonl         # Training (query, passage) pairs
│   │   ├── eval_queries.jsonl        # Held-out evaluation queries
│   │   ├── eval_corpus.jsonl         # Evaluation passage corpus
│   │   └── eval_qrels.json          # Query-relevance judgments
│   ├── evaluation/                    # Evaluation results
│   │   ├── baseline_results.json
│   │   ├── experiment_results.json
│   │   └── finmteb_results.json
│   └── sample/                        # Small sample for reviewers to run
│
├── src/
│   ├── data/
│   │   ├── esgbench_parser.py        # ESGBench → (query, passage) pairs
│   │   ├── cdp_parser.py             # CDP disclosures → pairs
│   │   ├── gri_parser.py             # GRI reports → pairs
│   │   ├── tcfd_parser.py            # TCFD reports → pairs
│   │   ├── synthetic_generator.py    # LLM-based query augmentation
│   │   ├── dataset_builder.py        # Merge, deduplicate, split, filter
│   │   └── schemas.py                # Pydantic models for data
│   ├── training/
│   │   ├── trainer.py                # Fine-tuning pipeline (SentenceTransformers)
│   │   ├── hard_negatives.py         # Hard negative mining
│   │   ├── config.py                 # Experiment hyperparameters
│   │   └── callbacks.py              # Custom training callbacks
│   ├── evaluation/
│   │   ├── baselines.py              # BM25 + off-the-shelf evaluation
│   │   ├── retrieval_eval.py         # IR metrics (NDCG, MRR, Recall)
│   │   ├── hybrid.py                 # BM25 + dense fusion
│   │   ├── ablation.py               # Training data size ablation
│   │   ├── finmteb_eval.py           # Generalisation check
│   │   ├── failure_analysis.py       # Per-query error categorisation
│   │   └── visualisation.py          # UMAP, cluster metrics, result plots
│   ├── search/
│   │   ├── index.py                  # Build search indices (BM25 + dense)
│   │   └── searcher.py               # Unified search interface
│   ├── pipeline.py                    # End-to-end orchestration
│   └── api.py                         # FastAPI endpoints
│
├── app/
│   └── streamlit_app.py               # Side-by-side search comparison UI
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Source data statistics, quality checks
│   ├── 02_dataset_construction.ipynb  # Building training pairs, visualising distribution
│   ├── 03_baseline_evaluation.ipynb   # BM25 and off-the-shelf results
│   ├── 04_training_experiments.ipynb  # Fine-tuning runs, hyperparameter analysis
│   ├── 05_results_analysis.ipynb      # Full evaluation results + visualisations
│   └── 06_embedding_analysis.ipynb    # UMAP, cluster metrics, before/after
│
├── tests/
│   ├── test_parsers.py               # Data parser unit tests
│   ├── test_dataset_builder.py       # Dataset construction tests
│   ├── test_evaluation.py            # Metric computation tests
│   ├── test_search.py                # Search interface tests
│   └── test_pipeline.py
│
└── docs/
    ├── architecture.md                # Architecture diagram + design decisions
    ├── dataset_construction.md        # Full details on training data pipeline
    ├── training_experiments.md        # All experiments with hyperparameters + results
    └── evaluation_results.md          # Full results + failure analysis
```

---

## 10. Communication Plan

### README Structure (< 500 words)

1. **One-liner:** "The first ESG-specific sentence-transformer for information retrieval — fine-tuned on 15K+ query-passage pairs from real sustainability reports, published to HuggingFace Hub."
2. **The problem:** General-purpose embedding models drop 20–40% on domain-specific retrieval (arXiv 2409.18511). Sustainability reports use specialised vocabulary (Scope 3, TCFD, GRI 305-1, science-based targets) that general models weren't trained on. ESG analytics platforms (MSCI, Clarity AI, Bloomberg) need retrieval that understands this domain.
3. **What this builds:** A sentence-transformer fine-tuned on ESG text using contrastive learning. Training data constructed from 4 sources (ESGBench, CDP, GRI, TCFD). Evaluated against BM25, off-the-shelf dense, and hybrid baselines using standard IR metrics. Published to HuggingFace Hub.
4. **Key results:** NDCG@10 improvement of [X]% over general embeddings. Fine-tuning helps most on [query type]. [X]K training pairs needed before fine-tuning outperforms baseline. (Numbers filled in after evaluation.)
5. **Try it:** Link to deployed Streamlit search comparison app. Link to HuggingFace model.
6. **Tech stack:** Python, PyTorch, SentenceTransformers, ONNX Runtime, rank-bm25, HuggingFace Hub, Streamlit, FastAPI, Docker.
7. **How to run:** `pip install -e . && python -m src.pipeline --sample` (runs on included sample data).

### Blog Post Outline

**Title:** "I Built an ESG Search Engine That Actually Understands Sustainability Reports"

1. **Hook:** "I asked a general-purpose embedding model to find information about Scope 3 emissions in a sustainability report. It ranked a passage about scope of operations higher than the actual emissions disclosure. So I built a model that knows the difference."
2. **The 20–40% gap:** Published research shows general embeddings underperform on domain text. ESG is worse than most domains because the vocabulary is specialised, acronym-heavy, and framework-specific (GRI, TCFD, SASB, CDP, CSRD, SBTi — and that's just the acronyms).
3. **Building a training dataset from scratch:** Parsing ESGBench, CDP disclosures, GRI reports, and TCFD frameworks into 15K+ (query, passage) pairs. Why this was harder than downloading a CSV. What each source contributed to the dataset.
4. **The training:** Contrastive learning with MultipleNegativesRankingLoss. Why hard negative mining matters for ESG text. The difference between "obviously wrong" and "subtly wrong" passages in sustainability reporting.
5. **Did it work?** NDCG@10 before and after. Which queries improved most. Where BM25 still wins. The training data ablation curve: how much ESG data you need before fine-tuning pays off.
6. **The embedding space, visualised:** UMAP before and after — backed by silhouette scores, not just pretty pictures.
7. **What I'd do next:** Larger training set (synthetic generation at scale), multi-lingual ESG embeddings (EU sustainability reports are in 24 languages), integration into a full RAG pipeline (connecting P2 to L1-style systems).
8. **Honest limitations:** English-only, Environmental topic bias, limited evaluation scale, no temporal evaluation (does the model work on 2025 reports if trained on 2020–2024 data?).

**Publish on:** Medium/TDS + LinkedIn.

### Interview Talking Points (STAR Format)

**S:** ESG analytics platforms (MSCI, Clarity AI, Bloomberg) need to search across thousands of sustainability reports using specialised vocabulary — Scope 3, TCFD, science-based targets. General-purpose embedding models drop 20–40% on domain-specific retrieval tasks. No ESG-specific sentence-transformer existed.

**T:** Build and publish the first ESG retrieval embedding model: construct a domain-specific training dataset, fine-tune a sentence-transformer with contrastive learning, rigorously evaluate against baselines, and publish the model to HuggingFace Hub.

**A:** Constructed 15K+ (query, passage) pairs from four ESG sources — ESGBench (real QA from 45 sustainability reports), CDP climate disclosures, GRI-structured reports, and TCFD frameworks — plus synthetic augmentation. Fine-tuned all-mpnet-base-v2 using MultipleNegativesRankingLoss with hard negative mining across 8–12 experiments on a Colab T4. Evaluated against four baselines (BM25 sparse, two off-the-shelf dense models, and hybrid retrieval) using NDCG@10, MRR, and Recall@K on a held-out evaluation set. Conducted training data size ablation, per-query failure analysis categorised by ESG topic and query type, embedding space visualisation with cluster metrics, and generalisation check against FinMTEB. Published model to HuggingFace Hub with full model card. Deployed side-by-side search comparison as Streamlit app.

**R:** Fine-tuned model achieved [X]% NDCG@10 improvement over general embeddings on ESG retrieval, with strongest gains on [query type] queries. Training data ablation showed fine-tuning outperforms baselines above [N]K pairs, with diminishing returns above [M]K. [Finding about which ESG topics benefit most]. [Finding about hybrid retrieval]. Published model has [N] downloads on HuggingFace Hub. The model is the first ESG-specific sentence-transformer for information retrieval.

---

## 11. Scope and Timeline

### Pre-Commitment: Empirical Tests (3–4 hours, before Week 1)

Before committing to the full project:

- [ ] **Q1 (1 hr):** Download ESGBench from GitHub. Parse the QA pairs. Verify format, count, quality. Check that questions map to retrievable passages. **Go/no-go threshold:** ≥ 800 usable QA pairs that can be converted to (query, passage) format.
- [ ] **Q2 (1 hr):** Access CDP Data Portal. Check whether structured disclosure responses are downloadable for free-tier accounts. Try to parse 10 company disclosures. **Go/no-go threshold:** Structured Q&A pairs extractable from CDP without manual effort per company. If CDP requires manual download per company, scope data augmentation more heavily toward synthetic generation.
- [ ] **Q3 (1–2 hr):** Quick feasibility test. Take 50 ESGBench (query, passage) pairs. Encode with all-mpnet-base-v2. Compute retrieval metrics (NDCG@10) on this small set. Compare against BM25 on the same queries. **Go/no-go threshold:** Off-the-shelf dense retrieval is imperfect (NDCG@10 < 0.9) — if it's already near-perfect, there's no room for fine-tuning to improve. BM25 should be competitive but beatable on at least some queries.

### Phase 1: Dataset Construction (Weeks 1–2)

- [ ] Build ESGBench parser: QA pairs → (query, passage) pairs with source metadata
- [ ] Build CDP parser: disclosure responses → (query, passage) pairs
- [ ] Build GRI report parser: section headers → queries, section content → passages
- [ ] Build TCFD parser: disclosure questions → queries, report sections → passages
- [ ] Implement quality filters: length constraints, deduplication, language check
- [ ] Merge all sources into unified training dataset
- [ ] Implement synthetic query generation pipeline (LLM augmentation)
- [ ] Generate synthetic queries for passages lacking natural query counterparts
- [ ] Split into train (80–85%) and eval (15–20%) sets
- [ ] Dataset quality audit: topic distribution, length distribution, source balance
- [ ] Run baseline evaluation: BM25, off-the-shelf mpnet, off-the-shelf MiniLM, hybrid
- [ ] **Milestone:** 10K+ training pairs, baseline evaluation complete. "Before" picture established.

### Phase 2: Fine-Tuning + Evaluation (Week 3)

- [ ] Set up training pipeline (SentenceTransformers, Colab T4)
- [ ] Experiment 1: MNR loss, default hyperparameters
- [ ] Experiment 2: MNR loss + hard negative mining
- [ ] Experiment 3: Learning rate sweep (with hard negatives)
- [ ] Experiment 4: Training data composition variants (sources)
- [ ] Experiment 5: Training data size ablation (1K, 2K, 5K, 10K, 15K)
- [ ] Experiment 6: Fine-tune all-MiniLM-L6-v2 for model size comparison
- [ ] Select best model by eval set NDCG@10
- [ ] Full evaluation: all metrics, all baselines, including hybrid retrieval
- [ ] Per-query failure analysis
- [ ] Training data ablation curve
- [ ] Embedding space visualisation (UMAP before/after + silhouette scores)
- [ ] FinMTEB generalisation check
- [ ] Inference latency benchmark: base vs. fine-tuned vs. BM25 (CPU and GPU)
- [ ] **Milestone:** Best fine-tuned model selected. Full evaluation results. Training curve plotted.

### Phase 3: Publication + Deployment (Week 4)

- [ ] Export best model to ONNX format, benchmark latency vs. PyTorch (Layer 6)
- [ ] Write HuggingFace model card (description, training, evaluation, limitations, usage)
- [ ] Push model + ONNX variant to HuggingFace Hub
- [ ] Build Streamlit app (side-by-side search comparison)
- [ ] Build FastAPI endpoints (/search, /compare)
- [ ] Dockerise
- [ ] Deploy to Render
- [ ] **Milestone:** Published model on HuggingFace. Live demo URL.

### Phase 4: Communication (Week 5)

- [ ] Write README (< 500 words)
- [ ] Write blog post
- [ ] Record 2-min demo video (screen capture of search comparison app)
- [ ] Prepare interview talking points
- [ ] Final code cleanup: docstrings, type hints, linting, tests
- [ ] **Milestone:** Published blog post. Complete repository.

### Total: 5 weeks

---

## 12. Estimated Costs

| Item | Cost |
|---|---|
| ESGBench | Free (GitHub) |
| CDP Data Portal | Free (registration required) |
| GRI reports | Free (company websites) |
| TCFD reports | Free (company websites) |
| Synthetic query generation (LLM API) | ~$5–15 |
| Google Colab T4 (training) | Free tier (~10–15 GPU-hours, within free limits) |
| HuggingFace Hub (model hosting) | Free |
| Render deployment | Free tier |
| **Total** | **~$5–15** |

**Note:** If Colab free tier is insufficient (session timeouts during longer experiments), Colab Pro ($12/month) provides guaranteed GPU access. Total cost would rise to ~$17–27 for one month.

---

## 13. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Training dataset too small for meaningful improvement | Medium | High | Target 10–20K pairs from 4+ sources. Synthetic augmentation fills gaps. Training data ablation shows the trajectory — even if current improvement is modest, the curve demonstrates that more data would help. |
| ESGBench too easy for BM25 (keyword overlap in structured Q&A) | Low-Medium | High | Include diverse query types in evaluation: paraphrase and conceptual queries where dense should outperform BM25. Report results disaggregated by query difficulty. If BM25 is near-perfect on structured queries, the fine-tuned model's value is on the harder queries. |
| CDP/GRI data harder to parse than expected | Medium | Medium | Pre-commitment test (Q2) catches this early. Fallback: lean more heavily on ESGBench + synthetic generation. Document parsing challenges honestly. |
| Fine-tuned model shows marginal improvement (<5% NDCG) | Medium | Medium | Per-query analysis shows *where* it helps (even if aggregate is small). Training curve shows trajectory. Hybrid retrieval may show larger gains. If improvement is genuinely negligible, "general models are adequate for ESG" is itself a finding — document honestly. |
| Project looks like a tutorial follow-along | Medium | High | The four-layer differentiation (ESG domain + novel dataset + rigorous evaluation + published artifact) pushes far beyond any tutorial. No tutorial exists for ESG retrieval embeddings. |
| Colab T4 insufficient for all experiments | Low | Low | Most experiments complete in 1–2 hours. Schedule experiments across sessions. Colab Pro ($12) as fallback. |
| HuggingFace model gets zero engagement | Low | Low | The model's portfolio value is in the process and evaluation, not in download count. But: tag the model with relevant keywords (ESG, sustainability, TCFD, GRI), write a blog post linking to it, and cross-post to relevant communities. |
| Environmental topic bias in training data | High | Medium | Expected and documented. CDP and TCFD are climate-focused. GRI covers broader ESG topics but Environmental is still dominant. Include topic distribution in dataset audit. Flag this as a known limitation and future work (social/governance augmentation). |

---

## 14. Limitations (Pre-Written)

These will appear in the README, blog post, model card, and evaluation documentation:

**Data limitations:**
- "Training data is heavily skewed toward Environmental topics (climate, emissions, energy) because CDP and TCFD frameworks are climate-focused. Social and Governance retrieval may not benefit equally from fine-tuning. Topic distribution: approximately [X]% Environmental, [Y]% Social, [Z]% Governance."
- "Training data is English-only. ESG reports from non-Anglophone companies (which may report in local languages or use different terminology conventions) are not represented. EU sustainability reporting under CSRD will produce reports in 24 languages — this model covers only one."
- "Training data spans [time period]. ESG reporting vocabulary evolves rapidly (CSRD replaces NFRD, ISSB introduces new standards, the EU Taxonomy adds new criteria). The model may not handle reporting language from standards adopted after the training data cutoff."
- "The evaluation corpus is relatively small (~150–200 queries). Results have moderate confidence intervals. Larger-scale evaluation on a broader ESG corpus would provide more reliable metrics."

**Model limitations:**
- "Fine-tuning a 110M parameter model on 15K pairs is at the lower end of what's typical for domain adaptation. The training data ablation curve shows whether the model is still improving at this scale — if so, more data would produce a better model."
- "The model is optimised for passage-level retrieval (100–500 word passages). Very short passages (single sentences) or very long documents (full reports) may not be well-served. In production, document chunking strategy matters as much as the embedding model."
- "No multi-lingual capability. A production ESG search system would need multi-lingual embeddings to handle the global reporting landscape."

**What a production system would do differently:**
- "With access to commercial ESG data providers (MSCI, Sustainalytics), the training set could include 100K+ high-quality pairs — dramatically stronger fine-tuning signal."
- "A production model would be continuously updated as new sustainability standards are adopted and reporting conventions evolve."
- "Active learning from user search feedback (which results were clicked, which were ignored) would improve the model iteratively without manual annotation."
- "Multi-lingual training would cover the 24 EU official languages, Chinese, Japanese, and other major reporting languages."

---

## 15. References and Prior Art to Cite

### Domain-Specific Embedding Research
- **"Do We Need Domain-Specific Embedding Models?"** (arXiv 2409.18511): Demonstrates 20–40% performance drop for general models on domain-specific tasks. Primary justification for the project.
- **FinMTEB** (arXiv, Feb 2025): Financial embedding benchmark, 64 datasets, 7 tasks. Used for generalisation testing.
- **MLEB: Massive Legal Embedding Benchmark** (arXiv): Legal domain equivalent. Shows the pattern of domain-specific benchmarks emerging across verticals.

### ESG NLP Models
- **ESG-BERT** (Mehra, 2022; GitHub): BERT for ESG text classification. Demonstrates domain adaptation value for ESG but solves classification, not retrieval.
- **ClimateBERT** (Webersinke et al., 2022; arXiv 2110.12010): Domain-adapted RoBERTa on climate text. Masked LM, not sentence-transformer.
- **FinBERT-ESG** (HuggingFace: yiyanghkust/finbert-esg): FinBERT variant for ESG sentiment. Classification, not retrieval.

### Datasets
- **ESGBench** (arXiv, Nov 2025; GitHub): QA benchmark for explainable ESG analysis. Primary seed dataset.
- **CDP** (cdp.net): Climate Disclosure Project structured disclosures.
- **GRI Standards** (globalreporting.org): Global Reporting Initiative sustainability reporting framework.
- **TCFD** (tcfdhub.org): Task Force on Climate-Related Financial Disclosures framework.

### Sentence-Transformer Training
- **SentenceTransformers documentation** (sbert.net): Loss overview, training overview, InformationRetrievalEvaluator.
- **HuggingFace blog:** "How to Train Sentence Transformers." Official fine-tuning guide.
- **Pinecone guide:** "Fine-tune Sentence Transformers with MNR Loss." Practical tutorial reference.

### Information Retrieval Evaluation
- **BEIR benchmark** (NeurIPS 2021): Standard IR benchmark. Establishes NDCG@10 as the primary metric.
- **rank-bm25** (PyPI): BM25 implementation for baseline comparison.
- **NanoBEIR** (SentenceTransformers): Lightweight evaluation for generalisation testing.

### Business Context
- **MSCI ESG Ratings methodology** (msci.com)
- **Clarity AI** (clarity.ai)
- **Bloomberg ESG module documentation**
- **EU Corporate Sustainability Reporting Directive (CSRD):** Regulatory context for ESG reporting expansion.

---

## 16. Checklist (From Portfolio Principles Reference)

### Business Framing
- [x] Clear business question in first sentence ("Can domain-specific embeddings improve ESG information retrieval?")
- [x] Identified stakeholder (ESG analysts, sustainability consultants, ESG analytics platforms)
- [x] Quantified cost of current solutions (MSCI $5K–2M/yr, Bloomberg ~$24K/yr)
- [x] Estimated business impact with stated assumptions (€350K–700K/yr recovered analyst productivity; Section 1)
- [x] Specific, actionable output (published HuggingFace model + evaluation results)

### Technical Execution
- [x] Real-world data (sustainability reports, CDP disclosures, GRI reports, TCFD frameworks)
- [x] Multiple data sources joined together (ESGBench + CDP + GRI + TCFD + synthetic)
- [x] Substantial data engineering (multi-source parser pipeline, quality filtering, deduplication)
- [x] Appropriate technique choice, justified (contrastive learning is the standard for embedding fine-tuning; domain justified by 20–40% performance gap)
- [x] Proper evaluation with multiple metrics (NDCG@10, MRR, Recall@K, ablation, failure analysis)
- [x] Honest limitation documentation (pre-written in Section 14)

### Modern Stack
- [x] Evaluation framework for model outputs (6-layer framework, Layers 0–6)
- [x] Cost and latency analysis planned (training compute cost, inference latency benchmark, ONNX export)
- [x] Failure mode documentation (per-query failure analysis categorised by topic/type)
- [x] Comparison against sensible baselines (BM25, off-the-shelf dense, hybrid)

### Deployment & Engineering
- [x] Interactive demo (Streamlit side-by-side search comparison)
- [x] API endpoint (FastAPI — /search, /compare)
- [x] Docker + cloud deployment (Render free tier)
- [x] Published artifact (HuggingFace model + ONNX variant)
- [x] Professional code structure (src/, tests/, docs/, notebooks/)
- [x] Version control with meaningful commit history planned

### Communication
- [x] README plan (< 500 words, business-first, with key findings)
- [x] Blog post outline ("I Built an ESG Search Engine That Actually Understands Sustainability Reports")
- [x] Resume line drafted
- [x] Interview talking points (STAR format)
- [x] Architecture diagram included

### Novelty
- [x] Searched HuggingFace, GitHub, Kaggle, Medium, arXiv
- [x] Framing distinct from existing projects (first ESG retrieval embedding model)
- [x] Combination of data sources/techniques is original (multi-source ESG dataset + contrastive learning + retrieval evaluation)
- [x] Not replicable by following a tutorial (no ESG embedding fine-tuning tutorial exists)
- [x] Prior art acknowledged and differentiation stated (Section 3)

---

*This document is the complete plan for P2. Implementation begins with the empirical pre-commitment tests (Section 11), followed by Phase 1 (Weeks 1–2).*
