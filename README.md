# MovieLens Recommender System

A modular, industry-standard recommender system built with PyTorch, following the classic **two-stage architecture** (Candidate Generation → Ranking) used at companies like Google, YouTube, and Netflix.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     All Movies (~3,700)                         │
│                          ↓                                      │
│  ┌──────────────────────────────────────────┐                   │
│  │  Stage 1: Candidate Generation           │                   │
│  │  Two-Tower Model (Implicit Feedback)     │                   │
│  │  → Retrieves Top-100 candidates          │                   │
│  │  → Optimized for RECALL                  │                   │
│  └──────────────────────────────────────────┘                   │
│                          ↓                                      │
│  ┌──────────────────────────────────────────┐                   │
│  │  Stage 2: Ranking                        │                   │
│  │  Deep Network (Hybrid: Explicit+Implicit)│                   │
│  │  → Scores & re-ranks candidates          │                   │
│  │  → Optimized for PRECISION               │                   │
│  └──────────────────────────────────────────┘                   │
│                          ↓                                      │
│               Top-10 Recommendations                            │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Choices

| Aspect | Candidate Generation | Ranking |
|--------|---------------------|---------|
| **Goal** | Recall (cast wide net) | Precision (perfect ordering) |
| **Feedback** | Implicit (binary: interacted or not) | Hybrid (explicit rating + implicit signal) |
| **Model** | Two-Tower (Dual Encoder) | Deep MLP with residual blocks |
| **Loss** | Contrastive (in-batch negatives) | Binary Cross-Entropy |
| **Negative Sampling** | In-batch (~1023 per positive) | Pre-sampled 1:4 popularity-weighted |

## Project Structure

```
MovieLens1/
├── configs/default.yaml       # All hyperparameters
├── src/
│   ├── data/                  # Download, preprocess, datasets
│   ├── features/              # Feature encoding & vocab management
│   ├── models/                # Two-Tower & Ranking models
│   ├── training/              # Trainer, loss functions
│   ├── evaluation/            # Retrieval & ranking metrics
│   ├── inference/             # End-to-end recommendation pipeline
│   └── utils/                 # Config, seeding, logging helpers
├── scripts/                   # Individual training & eval scripts
├── main.py                    # CLI entry point
└── requirements.txt           # Dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python main.py download
```

### 3. Preprocess Data

```bash
python main.py preprocess
```

### 4. Train Models

```bash
# Train candidate generation (Two-Tower)
python main.py train --stage candidate_gen

# Train ranking model
python main.py train --stage ranking
```

### 5. Evaluate

```bash
python main.py evaluate
```

### 6. Get Recommendations

```bash
python main.py recommend --user_id 42
python main.py recommend --user_id 42 --top_n 20
```

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

- **Embedding dimensions**: 32 for IDs, 4-16 for categorical features
- **Candidate gen**: 64-d shared embedding space, temperature=0.05, batch_size=1024
- **Ranking**: 256→128→64 MLP with residual connections, dropout=0.2
- **Negative sampling**: 1:4 ratio, popularity-weighted
- **Training**: Adam optimizer, ReduceLROnPlateau scheduler, gradient clipping, early stopping

## Dataset

**MovieLens-1M**: ~1 million ratings from 6,040 users on 3,706 movies.
Data is split temporally (80/10/10 train/val/test) to avoid data leakage.
