# MovieLens Recommender System

A modular, industry-standard recommender system built with PyTorch, following the classic **two-stage architecture** (Candidate Generation → Ranking)

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
| **Loss** | In-batch softmax contrastive | Binary Cross-Entropy |
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


## Notes

Global temporal split is used over leave-last-out split to prevent potential data leakage.

There are 3 categories of samples:
1. Observed positive samples: Interactions with rating >= min_rating
2. Observed negative samples: Interactions with rating < min_rating
3. Synthetic negative samples: Randomly sampled from the set of movies the user hasn't interacted with, weighted by movie popularity. 

The synthetic negative samples have a 4:1 ratio with all observed samples (positive and negative) meaning that the the true ratio of negative to positive samples is closer to 5:1.

Candidate Generation uses a Two-Tower model with in-batch negatives. In-batch negatives are computationally cheap and provide massive scale, which suits the recall-focused goal of this stage.
Ranking uses a deep MLP with residual connections and popularity-weighted pre-sampled negatives. Hard negatives (popular unseen movies) force the model to learn fine-grained user preferences, which suits the precision-focused goal of this stage.

It is ensured that all feature vocabs (gender, age, occupation) are seen during training, but not all movie IDs are guaranteed to appear in the training set due to the temporal split. Movies only rated after the training cutoff have no learned embedding — this is the cold-start problem. However, this is partially mitigated by our hybrid approach, where content-based features (genres, user demographics) still provide a meaningful representation even when the movie ID embedding falls back to the padding index (0).

When measuring the loss of our candidate generation model, we use in-batch softmax contrastive loss. This means that for each user in the batch, we treat the other users' items as negatives. This is computationally cheap and provides massive scale, which suits the recall-focused goal of this stage as we're okay with False Negatives. Our logits are the similarity matrix between users and items which is temperature-scaled. Cross-entropy is applied row-wise (each user classifies its correct item among all batch items) and column-wise via the transpose (each item classifies its correct user). We average both directions to learn a symmetric embedding space.

We use BCEwithlogits loss for our ranking model. This is a standard binary cross-entropy loss that is applied to the predicted scores of the ranking model. It is a good choice for this task because it is a differentiable loss function that can be optimized using gradient descent. The labels are the tricky part because we want to be very accurate with our predictions. Our current min_rating is 4 which means that any rating 4 and higher are treated as positives. Any rating below that are negative. We also sample negatives with a 4:1 ratio to positives on the unrated movies. The biggest gray area is how we treat 3s as they are the most ambigious.

Original training/eval loop for the candidate generation model calculated the dot product similiarity matrix between all users and items which was very memory intensive. This was changed to chunked recall at k which processes users in chunks of `chunk_size`, keeping memory at O(chunk_size * N) instead of O(N^2). torch.topk is used to get the top k items for each user. This is efficient because it only keeps the top k items in memory at a time giving us a time complexity of O(chunk_size * N * log k) instead of O(N^2).

LayerNorm or no normalization is preferred for NLP and embedding models because embeddings must be consistent and comparable across samples.
BatchNorm introduces dependency on batch statistics, causing embeddings to vary across batches, which destabilizes similarity learning.