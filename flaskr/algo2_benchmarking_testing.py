import random
import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Random seed set to {SEED}")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    print("No GPU found — running on CPU.")

print(f"Device: {DEVICE}")


# --- Sequence ---
MAX_SEQ_LEN  = 100     # increased from 50: median user history is 60, 75th pct is 133

# --- Model architecture ---
D_MODEL      = 64       # reduced from 128: only 485 users, smaller model avoids overfitting
NUM_HEADS    = 2       # number of self-attention heads (D_MODEL must be divisible)
NUM_BLOCKS   = 2       # reduced from 3: fewer layers suit small-medium datasets
DROPOUT      = 0.3     # increased from 0.2: stronger regularisation for small dataset

# --- Training ---
LR           = 1e-3    # Adam learning rate
EPOCHS       = 50      # maximum training epochs
PATIENCE     = 5       # early stopping patience (epochs without HR@10 improvement)
BATCH_SIZE   = 64     # reduced from 256: more gradient steps per epoch (~4 batches vs ~2)
NEG_SAMPLES  = 20      # increased from 4: more negatives sharpen ranking in sampled softmax CE
TEMPERATURE  = 0.1     # reduced from 0.07: 0.07 too aggressive for recsys scale, saturates early

# --- Evaluation ---
TOP_K        = 10      # cut-off for HR@K and NDCG@K
ALPHA = 0.65   # weight for nDCG in composite score

# --- Reproducibility ---
RANDOM_STATE = SEED

print("Hyperparameters:")
for name, val in [
    ("MAX_SEQ_LEN", MAX_SEQ_LEN), ("D_MODEL", D_MODEL), ("NUM_HEADS", NUM_HEADS),
    ("NUM_BLOCKS", NUM_BLOCKS), ("DROPOUT", DROPOUT), ("LR", LR),
    ("EPOCHS", EPOCHS), ("PATIENCE", PATIENCE), ("BATCH_SIZE", BATCH_SIZE),
    ("NEG_SAMPLES", NEG_SAMPLES), ("TOP_K", TOP_K), ("TEMPERATURE", TEMPERATURE),
]:
    print(f"  {name:<15} = {val}")


DATA_DIR = os.path.join(os.path.abspath(".."), "rec","dataset")

# Keep timestamp — critical for building ordered sequences
ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
movies_df  = pd.read_csv(os.path.join(DATA_DIR, "movie_info.csv"))
genre_df = pd.read_csv(os.path.join(DATA_DIR, "genre.csv"), sep="|", header=None, names=["genre", "genre_id"])
genre_dict = dict(zip(genre_df["genre"], genre_df["genre_id"]))

print(genre_dict)
print(f"Ratings : {ratings_df.shape}  |  columns: {list(ratings_df.columns)}")
print(f"Movies  : {movies_df.shape}   |  columns: {list(movies_df.columns)}")
print(f"Genre  : {genre_df.shape}   |  columns: {list(genre_df.columns)}")

def encode_genres(genre_string):
    # If NaN → return [19]
    if pd.isna(genre_string):
        return [genre_dict["(no genres listed)"]]
    genre_string = genre_string.strip()

    # If empty string → return [19]
    if genre_string == "":
        return [genre_dict["(no genres listed)"]]
    genres = genre_string.split("|")
    
    # If explicitly "(no genres listed)"
    if "(no genres listed)" in genres:
        return [genre_dict["(no genres listed)"]]
    
    # Normal case
    return [genre_dict[g] for g in genres]

movies_df["genre_ids"] = movies_df["genres"].apply(encode_genres)
movies_df["genre_ids"] = movies_df["genre_ids"].apply(
    lambda x: x if isinstance(x, list) else [genre_dict["(no genres listed)"]]
)
print(movies_df[["movieId", "genre_ids"]].head())
ratings_df = ratings_df.merge(
    movies_df[["movieId", "title", "genre_ids"]],
    on="movieId",
    how="left"
)

user_enc = LabelEncoder()
item_enc = LabelEncoder()

ratings_df["user_idx"] = user_enc.fit_transform(ratings_df["userId"])
ratings_df["item_idx"] = item_enc.fit_transform(ratings_df["movieId"]) + 1  # 0 = padding
ratings_df = ratings_df.dropna(subset=["title"])

NUM_USERS = int(ratings_df["user_idx"].nunique())
NUM_ITEMS = int(ratings_df["item_idx"].max()) + 1  # +1 for padding index 0

print(f"Unique users : {NUM_USERS}")
print(f"Unique items : {NUM_ITEMS - 1}  (vocab size incl. pad token: {NUM_ITEMS})")
print(ratings_df.head(3))

# Sort globally by user then timestamp, then group into per-user sequences
ratings_sorted = ratings_df.sort_values(["user_idx", "timestamp"])

user_sequences = (
    ratings_sorted
    .groupby("user_idx")[["item_idx", "genre_ids", "rating", "title"]]
    .apply(lambda x: x.values.tolist())
)
lengths = [len(v) for v in user_sequences.values]
print(f"Users with sequences : {len(user_sequences)}")
print(f"Sequence length  -  min: {min(lengths)},  max: {max(lengths)},  "
      f"median: {int(np.median(lengths))},  mean: {np.mean(lengths):.1f}")
print(user_sequences[0][:5]) # [movie idx, [genre_ids], rating, title]


# def pad_or_truncate(seq, max_len, pad_val=0):
#     """Left-pad with pad_val, or truncate to keep the most recent max_len items."""
#     seq = list(seq)
#     if len(seq) >= max_len:
#         return seq[-max_len:]
#     return [pad_val] * (max_len - len(seq)) + seq

def pad_list(seq, max_len, pad_val):
    """Left-pad with pad_val, or truncate to keep the most recent max_len items."""
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_val] * (max_len - len(seq)) + seq

# train_seqs     = []   # padded input sequences          shape (N, MAX_SEQ_LEN)
# train_pos      = []   # padded positive next-item seqs  shape (N, MAX_SEQ_LEN)
# val_seqs       = []   # padded input sequences for val  shape (M, MAX_SEQ_LEN)
# val_targets    = []   # single target item per val user  shape (M,)
# user_histories = {}   # user_idx -> set of all interacted item indices (for neg. sampling)

# Collect all genre ids
all_genres = set()

for seq in user_sequences.values:   # ✅ iterate over Series values
    for item, genre_list, rating, title in seq:
        all_genres.update(genre_list)

NUM_GENRES = max(all_genres) + 1   # assuming 0-based ids
print("Number of genres:", NUM_GENRES)

def to_multihot(genre_list, num_genres):
    vec = np.zeros(num_genres, dtype=np.float32)
    for g in genre_list:
        vec[g] = 1.0
    return vec
    
train_rows = []
val_rows = []
user_histories = {}   # user_idx -> set of all interacted item indices (for neg. sampling)

for user_idx, seq in user_sequences.items():

    if len(seq) < 3:
        continue

    # Split into separate lists
    items   = [x[0] for x in seq]
    genres  = [to_multihot(x[1], NUM_GENRES) for x in seq]
    ratings = [x[2] for x in seq]
    user_histories[user_idx] = set(items)

    # Validation
    val_rows.append({
        "user_idx": user_idx,
        "input_items": items[:-1],
        "input_genres": genres[:-1],
        "input_ratings": ratings[:-1],
        "target_item": items[-1]
    })

    # Training
    train_items = items[:-1]

    train_rows.append({
        "user_idx": user_idx,
        "input_items": train_items[:-1],
        "pos_items": train_items[1:],
        "input_genres": genres[:-2],
        "input_ratings": ratings[:-2],
    })

train_df = pd.DataFrame(train_rows)
val_df   = pd.DataFrame(val_rows)
zero_genre = np.zeros(NUM_GENRES, dtype=np.float32)

train_df["input_items_pad"] = train_df["input_items"].apply(
    lambda x: pad_list(x, MAX_SEQ_LEN, 0)
)
train_df["pos_items_pad"] = train_df["pos_items"].apply(
    lambda x: pad_list(x, MAX_SEQ_LEN, 0)
)
train_df["input_ratings_pad"] = train_df["input_ratings"].apply(
    lambda x: pad_list(x, MAX_SEQ_LEN, 0.0)
)
train_df["input_genres_pad"] = train_df["input_genres"].apply(
    lambda x: pad_list(x, MAX_SEQ_LEN, zero_genre)
)


val_df["input_items_pad"] = val_df["input_items"].apply(
    lambda x: pad_list(x, MAX_SEQ_LEN, 0)
)
val_df["input_ratings_pad"] = val_df["input_ratings"].apply(
    lambda x: pad_list(x, MAX_SEQ_LEN, 0.0)
)
val_df["input_genres_pad"] = val_df["input_genres"].apply(
    lambda x: pad_list(x, MAX_SEQ_LEN, zero_genre)
)


print(val_df.columns)

train_items = np.array(train_df["input_items_pad"].tolist(), dtype=np.int64)
train_pos   = np.array(train_df["pos_items_pad"].tolist(), dtype=np.int64)
train_genres = np.array(train_df["input_genres_pad"].tolist(), dtype=np.float32)
train_ratings = np.array(train_df["input_ratings_pad"].tolist(), dtype=np.float32)

val_items = np.array(val_df["input_items_pad"].tolist(), dtype=np.int64)
val_targets   = np.array(val_df["target_item"].tolist(), dtype=np.int64)
val_genres = np.array(val_df["input_genres_pad"].tolist(), dtype=np.float32)
val_ratings = np.array(val_df["input_ratings_pad"].tolist(), dtype=np.float32)

print(f"Training samples   : {train_df.shape}")
print(f"Validation samples : {val_df.shape}")
print(f"Users skipped (seq < 3): {len(user_sequences) - len(val_df)}")


# Fraction of non-padding positions (higher = denser sequences)
train_density = (train_pos != 0).sum() / train_pos.size
val_density   = (val_items  != 0).sum() / val_items.size

print("=" * 45)
print("  Dataset Summary")
print("=" * 45)
print(f"  Users (train/val)  : {len(train_items):>7,}")
print(f"  Item vocabulary    : {NUM_ITEMS:>7,}  (incl. pad)")
print(f"  Sequence length    : {MAX_SEQ_LEN:>7}")
print(f"  Train non-pad ratio: {train_density:>7.1%}")
print(f"  Val   non-pad ratio: {val_density:>7.1%}")
print("=" * 45)
print(f"\nSample train input  : {train_items[0]}")
print(f"Sample train labels : {train_pos[0]}")
print(f"Sample val target   : {val_targets[0]}")




import math
import torch.nn as nn
import torch.nn.functional as F

# ── Point-wise Feed-Forward ──────────────────────────────────────────────────
class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_model * 4)
        self.fc2     = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# ── Single Transformer Block ─────────────────────────────────────────────────
class SASRecBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn  = PointWiseFeedForward(d_model, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention with pre-LN + residual
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x,
                         attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = self.drop(x) + residual
        # FFN with pre-LN + residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x) + residual
        return x


# ── SASRec ───────────────────────────────────────────────────────────────────
class SASRec(nn.Module):
    def __init__(self, num_items, num_genres, d_model, num_heads, num_blocks,
                 max_seq_len, dropout=0.2):
        super().__init__()
        self.item_emb    = nn.Embedding(num_items, d_model, padding_idx=0)
        self.pos_emb     = nn.Embedding(max_seq_len + 1, d_model)   # 1-indexed

        # === NEW ===
        self.genre_proj  = nn.Linear(num_genres, d_model)
        self.rating_proj = nn.Linear(1, d_model)
        self.fusion_gate = nn.Linear(d_model * 3, d_model)
        self.sigmoid = nn.Sigmoid()
        self.dropout2     = nn.Dropout(dropout)
        self.ln      = nn.LayerNorm(d_model)
        self.out_proj  = nn.Linear(d_model*3, d_model)

        self.dropout     = nn.Dropout(dropout)
        
        self.blocks      = nn.ModuleList([
            SASRecBlock(d_model, num_heads, dropout) for _ in range(num_blocks)
        ])
        self.ln_out      = nn.LayerNorm(d_model)
        self.max_seq_len = max_seq_len
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,  std=0.02)
        nn.init.normal_(self.genre_proj.weight,  std=0.02)
        nn.init.normal_(self.rating_proj.weight,  std=0.02)
        nn.init.normal_(self.fusion_gate.weight,  std=0.02)

    def forward(self, seq, genres=None, ratings=None):
        """seq : (B, L) — item indices, 0 = padding → returns (B, L, D)"""
        B, L = seq.shape
        positions = torch.arange(1, L + 1, device=seq.device).unsqueeze(0).expand(B, -1)

        
        item_emb = self.item_emb(seq)
        pos_emb  = self.pos_emb(positions)

        # x = item_emb + pos_emb

        # # ---- Add genre embedding ----
        # if genres is not None:
        #     # print(genres.size(), ratings.size(), seq.size())
        #     genre_emb = self.genre_proj(genres)   # (B, L, D)
        #     x = x + genre_emb

        # # ---- Add rating embedding ----
        # if ratings is not None:
        #     ratings = ratings.unsqueeze(-1)       # (B, L, 1)
        #     rating_emb = self.rating_proj(ratings)
        #     x = x + rating_emb
        # x = self.ln(x)

        base = item_emb 
        if genres is not None and ratings is not None:
            genre_emb  = self.genre_proj(genres)
            rating_emb = self.rating_proj(ratings.unsqueeze(-1))

            content = genre_emb + rating_emb
            content = self.dropout2(content)
            # gate decides how much content to use
            gate_input = torch.cat([base, content, base * content], dim=-1)
            gate = self.sigmoid(self.fusion_gate(gate_input))

            x = self.ln(base + gate * content ) + pos_emb
        else:
            x = base + pos_emb

        x = self.dropout(x)

        # Causal (autoregressive) mask — upper-triangular True = blocked
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )
        pad_mask = (seq == 0)   # (B, L), True where padding

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=pad_mask)

        return self.ln_out(x)   # (B, L, D)

    def predict(self, seq, candidate_items, genres=None, ratings=None):
        """
        seq             : (B, L)
        candidate_items : (B, C)
        returns scores  : (B, C)  — cosine similarities in [-1, 1]
        """
        # print(seq.size())
        h_last   = F.normalize(self.forward(seq, genres, ratings)[:, -1, :], dim=-1)  # (B, D) unit vector
        cand_emb = F.normalize(self.item_emb(candidate_items), dim=-1)  # (B, C, D) unit vectors
        return torch.bmm(cand_emb, h_last.unsqueeze(-1)).squeeze(-1)     # (B, C) cosine sim
print("SASRec model class defined")

class GRU4Rec(nn.Module):
    def __init__(self, num_items, num_genres, hidden_size,
                 num_layers=1, dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size

        # Embeddings
        self.item_emb = nn.Embedding(num_items, hidden_size, padding_idx=0)

        # Content fusion
        self.genre_proj  = nn.Linear(num_genres, hidden_size)
        self.rating_proj = nn.Linear(1, hidden_size)
        self.fusion_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.sigmoid     = nn.Sigmoid()

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.genre_proj.weight, std=0.02)
        nn.init.normal_(self.rating_proj.weight, std=0.02)
        nn.init.normal_(self.fusion_gate.weight, std=0.02)

    def forward(self, seq, genres=None, ratings=None):
        """
        seq: (B, L)
        returns: (B, L, H)
        """
        item_emb = self.item_emb(seq)
        base = item_emb

        if genres is not None and ratings is not None:
            genre_emb  = self.genre_proj(genres)
            rating_emb = self.rating_proj(ratings.unsqueeze(-1))
            content = genre_emb + rating_emb

            gate_input = torch.cat([base, content, base * content], dim=-1)
            gate = self.sigmoid(self.fusion_gate(gate_input))

            x = base + gate * content
            # x = base + genre_emb + rating_emb
        else:
            x = base

        x = self.dropout(x)

        output, _ = self.gru(x)   # (B, L, H)
        return output

    def predict(self, seq, candidate_items, genres=None, ratings=None):
        """
        Same interface as SASRec
        """
        h = self.forward(seq, genres, ratings)

        # Use LAST non-pad position
        mask = (seq != 0).unsqueeze(-1)
        h = h * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        h_last = h.sum(dim=1) / lengths

        h_last = F.normalize(h_last, dim=-1)
        cand_emb = F.normalize(self.item_emb(candidate_items), dim=-1)

        return torch.bmm(cand_emb, h_last.unsqueeze(-1)).squeeze(-1)







model = SASRec(
    num_items   = NUM_ITEMS,
    num_genres=NUM_GENRES,
    d_model     = D_MODEL,
    num_heads   = NUM_HEADS,
    num_blocks  = NUM_BLOCKS,
    max_seq_len = MAX_SEQ_LEN,
    dropout     = DROPOUT,
).to(DEVICE)



total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print()
print(model)



from torch.utils.data import Dataset, DataLoader

print("Hyperparameters:")
for name, val in [
    ("MAX_SEQ_LEN", MAX_SEQ_LEN), ("D_MODEL", D_MODEL), ("NUM_HEADS", NUM_HEADS),
    ("NUM_BLOCKS", NUM_BLOCKS), ("DROPOUT", DROPOUT), ("LR", LR),
    ("EPOCHS", EPOCHS), ("PATIENCE", PATIENCE), ("BATCH_SIZE", BATCH_SIZE),
    ("NEG_SAMPLES", NEG_SAMPLES), ("TOP_K", TOP_K), ("TEMPERATURE", TEMPERATURE),
]:
    print(f"  {name:<15} = {val}")
print("-" * 90)


# ── Dataset: sample NEG_SAMPLES negatives per position ───────────────────────
class SASRecDataset(Dataset):
    def __init__(self, seqs, pos_seqs, genres, ratings, user_id_list, user_histories, num_items):
        self.seqs           = seqs
        self.pos_seqs       = pos_seqs
        self.user_id_list   = user_id_list
        self.user_histories = user_histories
        self.num_items      = num_items
        self.genres = genres
        self.ratings = ratings

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        pos = self.pos_seqs[idx]
        history = self.user_histories.get(self.user_id_list[idx], set())
        
        # NEG_SAMPLES negatives per position → shape (L, NEG_SAMPLES)
        neg = np.zeros((len(pos), NEG_SAMPLES), dtype=np.int64)
        for t in range(len(pos)):
            if pos[t] == 0:
                continue
            sampled = []
            while len(sampled) < NEG_SAMPLES:
                j = random.randint(1, self.num_items - 1)
                while j in history:
                    j = random.randint(1, self.num_items - 1)
                sampled.append(j)
            neg[t] = sampled
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
            torch.tensor(self.genres[idx], dtype=torch.float32),
            torch.tensor(self.ratings[idx], dtype=torch.float32),
        )


# ── Derive per-user training targets ─────────────────────────────────────────
train_targets = np.array([
    pos[pos != 0][-1] if (pos != 0).any() else 0
    for pos in train_pos
], dtype=np.int64)


# ── Composite score ───────────────────────────────────────────────────────────
def composite_score(hr, ndcg, alpha=ALPHA):
    return alpha * ndcg + (1 - alpha) * hr


# ── Combined evaluation: HR@K, NDCG@K, BCE loss, Composite ───────────────────
# predict() now returns cosine similarities — no need to rescale by sqrt(D_MODEL)
def evaluate_model(model, seqs, targets, genres, ratings, user_id_list,
                   user_histories, num_items, top_k, device, num_neg=99,
                   seed=SEED):
    model.eval()
    hr_list, ndcg_list, loss_list = [], [], []
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(seed)

    with torch.no_grad():
        for i in range(len(seqs)):
            target = int(targets[i])
            if target == 0:
                continue
            history = user_histories.get(user_id_list[i], set())

            negs = []
            while len(negs) < num_neg:
                j = rng.randint(1, num_items - 1)
                if j not in history and j != target:
                    negs.append(j)
                    
            seq    = torch.tensor(seqs[i], dtype=torch.long).unsqueeze(0).to(device)
            genre_seq  = torch.tensor(genres[i], dtype=torch.float32).unsqueeze(0).to(device)
            rating_seq  = torch.tensor(ratings[i], dtype=torch.float32).unsqueeze(0).to(device)

            # genre_seq = None
            # rating_seq = None
            rating_seq = rating_seq / 5.0

            cands  = torch.tensor([target] + negs, dtype=torch.long).unsqueeze(0).to(device)
            scores = model.predict(seq, cands, genre_seq, rating_seq).squeeze(0)   # cosine sim, no extra scaling

            labels    = torch.zeros(1 + num_neg, device=device)
            labels[0] = 1.0
            l = bce(scores, labels).item()
            if not math.isnan(l):
                loss_list.append(l)

            _, top_indices = scores.topk(top_k)
            rank_tensor = (top_indices == 0).nonzero(as_tuple=True)[0]
            if len(rank_tensor):
                r = rank_tensor[0].item() + 1
                hr_list.append(1.0)
                ndcg_list.append(1.0 / math.log2(r + 1))
            else:
                hr_list.append(0.0)
                ndcg_list.append(0.0)

    hr   = float(np.mean(hr_list))   if hr_list   else 0.0
    ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    loss = float(np.mean(loss_list)) if loss_list else float("nan")
    comp = composite_score(hr, ndcg)
    return hr, ndcg, loss, comp


# ── Build ordered user-id list ────────────────────────────────────────────────
user_id_list = [uid for uid, seq in user_sequences.items() if len(seq) >= 3]

dataset = SASRecDataset(train_items, train_pos, train_genres, train_ratings, user_id_list, user_histories, NUM_ITEMS)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                     num_workers=0, pin_memory=(DEVICE.type == "cuda"))

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98))

# ── Scheduler: halve LR after 5 epochs without comp improvement ──────────────
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

WARMUP_EPOCHS = 10   # early stopping won't trigger before this epoch
ES_PATIENCE   = 10   # early stopping patience
LR_PATIENCE   = 5    # already set in scheduler above

best_comp   = 0.0
patience_ct = 0
history_log = {
    "epoch"     : [],
    "train_loss": [], "train_hr": [], "train_ndcg": [], "train_comp": [],
    "val_loss"  : [], "val_hr"  : [], "val_ndcg"  : [], "val_comp"  : [],
}

# Header
print(f"{'Epoch':>5}  {'LR':>8}  "
      f"{'TrLoss':>7}  {'TrHR':>6}  {'TrNDCG':>7}  {'TrComp':>7}  "
      f"{'VaLoss':>7}  {'VaHR':>6}  {'VaNDCG':>7}  {'VaComp':>7}")
print("-" * 90)


# for epoch in range(1, EPOCHS + 1):

#     model.train()
#     running_loss = 0.0

#     for seq, pos, neg, genres, ratings in loader:

#         seq, pos, neg = seq.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
#         genres, ratings = genres.to(DEVICE), ratings.to(DEVICE)
#         # genres, ratings = None, None
#         optimizer.zero_grad()
#         ratings = ratings / 5.0
#         B, L = seq.shape

#         # Forward
#         h = model(seq, genres, ratings)              # (B, L, H)
#         h_n = F.normalize(h, dim=-1)

#         pos_emb = F.normalize(model.item_emb(pos), dim=-1)
#         neg_emb = F.normalize(model.item_emb(neg), dim=-1)

#         # Cosine scores
#         pos_scores = (h_n * pos_emb).sum(-1, keepdim=True) / TEMPERATURE
#         neg_scores = torch.einsum('bld,blnd->bln', h_n, neg_emb) / TEMPERATURE

#         logits  = torch.cat([pos_scores, neg_scores], dim=-1)
#         targets = torch.zeros(B, L, dtype=torch.long, device=DEVICE)
#         mask    = (pos != 0)

#         loss = F.cross_entropy(logits[mask], targets[mask])

#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()

#         running_loss += loss.item()

#     train_ce   = running_loss / len(loader)
#     current_lr = optimizer.param_groups[0]["lr"]

#     # ───────── Evaluate ─────────
#     tr_hr, tr_ndcg, tr_loss, tr_comp = evaluate_model(
#         model, train_items, train_targets,
#         train_genres, train_ratings,
#         user_id_list, user_histories,
#         NUM_ITEMS, TOP_K, DEVICE
#     )

#     va_hr, va_ndcg, va_loss, va_comp = evaluate_model(
#         model, val_items, val_targets,
#         val_genres, val_ratings,
#         user_id_list, user_histories,
#         NUM_ITEMS, TOP_K, DEVICE
#     )

#     scheduler.step(va_comp)

#     is_best  = va_comp > best_comp
#     ckpt_msg = "  best" if is_best else ""

#     print(f"{epoch:>5}  {current_lr:>8.6f}  "
#           f"{train_ce:>7.4f}  {tr_hr:>6.4f}  {tr_ndcg:>7.4f}  {tr_comp:>7.4f}  "
#           f"{va_loss:>7.4f}  {va_hr:>6.4f}  {va_ndcg:>7.4f}  {va_comp:>7.4f}"
#           f"{ckpt_msg}")

#     if is_best:
#         best_comp   = va_comp
#         patience_ct = 0
#         torch.save(model.state_dict(), "best_gru4rec.pt")
#     else:
#         patience_ct += 1
#         if epoch > WARMUP_EPOCHS and patience_ct >= ES_PATIENCE:
#             print(f"\nEarly stopping at epoch {epoch}  "
#                   f"(best Comp = {best_comp:.4f})")
#             break

# print(f"\nTraining complete. Best Val Composite: {best_comp:.4f}")

for epoch in range(1, EPOCHS + 1):
    # ── Training pass: cosine similarity + temperature-scaled softmax CE ──────
    model.train()
    running_loss = 0.0

    for seq, pos, neg, genres, ratings in loader:
        seq, pos, neg, genres, ratings = seq.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE), genres.to(DEVICE), ratings.to(DEVICE)
        # seq: (B, L), pos: (B, L), neg: (B, L, NEG_SAMPLES)
        optimizer.zero_grad()
        genres = None
        ratings = None
        # ratings = ratings / 5.0


        B, L = seq.shape
        h       = model(seq, genres, ratings)                                         # (B, L, D) 
        h_n     = F.normalize(h, dim=-1)                            # (B, L, D) unit vectors
        pos_emb = F.normalize(model.item_emb(pos), dim=-1)          # (B, L, D) unit vectors
        neg_emb = F.normalize(model.item_emb(neg), dim=-1)          # (B, L, K, D) unit vectors

        # Cosine similarities scaled by temperature
        pos_scores = (h_n * pos_emb).sum(-1, keepdim=True) / TEMPERATURE        # (B, L, 1)
        neg_scores = torch.einsum('bld,blnd->bln', h_n, neg_emb) / TEMPERATURE  # (B, L, K)

        # Sampled softmax CE: positive is always class index 0
        logits  = torch.cat([pos_scores, neg_scores], dim=-1)        # (B, L, 1+K)
        targets = torch.zeros(B, L, dtype=torch.long, device=DEVICE)
        mask    = (pos != 0)                                         # (B, L)

        loss = F.cross_entropy(logits[mask], targets[mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

    train_ce   = running_loss / len(loader)
    current_lr = optimizer.param_groups[0]["lr"]

    # ── Evaluate both splits ──────────────────────────────────────────────────
    tr_hr, tr_ndcg, tr_loss, tr_comp = evaluate_model(
        model, train_items, train_targets, train_genres, train_ratings, user_id_list,
        user_histories, NUM_ITEMS, TOP_K, DEVICE
    )
    va_hr, va_ndcg, va_loss, va_comp = evaluate_model(
        model, val_items, val_targets, val_genres, val_ratings, user_id_list,
        user_histories, NUM_ITEMS, TOP_K, DEVICE
    )

    # Log
    history_log["epoch"     ].append(epoch)
    history_log["train_loss"].append(train_ce)
    history_log["train_hr"  ].append(tr_hr)
    history_log["train_ndcg"].append(tr_ndcg)
    history_log["train_comp"].append(tr_comp)
    history_log["val_loss"  ].append(va_loss)
    history_log["val_hr"    ].append(va_hr)
    history_log["val_ndcg"  ].append(va_ndcg)
    history_log["val_comp"  ].append(va_comp)

    # ── Step LR scheduler ─────────────────────────────────────────────────────
    scheduler.step(va_comp)

    # ── Checkpoint & early stopping ───────────────────────────────────────────
    is_best  = va_comp > best_comp
    ckpt_msg = "  best" if is_best else ""

    print(f"{epoch:>5}  {current_lr:>8.6f}  "
          f"{train_ce:>7.4f}  {tr_hr:>6.4f}  {tr_ndcg:>7.4f}  {tr_comp:>7.4f}  "
          f"{va_loss:>7.4f}  {va_hr:>6.4f}  {va_ndcg:>7.4f}  {va_comp:>7.4f}"
          f"{ckpt_msg}")

    if is_best:
        best_comp   = va_comp
        patience_ct = 0
        torch.save(model.state_dict(), "best_sasrec.pt")
    else:
        patience_ct += 1
        # Early stopping only triggers after warmup period
        if epoch > WARMUP_EPOCHS and patience_ct >= ES_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}  "
                  f"(best Comp = {best_comp:.4f}, no improvement for {ES_PATIENCE} epochs)")
            break

print(f"\nTraining complete.  Best Val Composite: {best_comp:.4f}")



model.load_state_dict(torch.load("best_sasrec.pt", map_location=DEVICE))

hr_final, ndcg_final, loss_final, comp_final = evaluate_model(   # ← 4 values
    model, val_items, val_targets, val_genres, val_ratings, user_id_list,
    user_histories, NUM_ITEMS, TOP_K, DEVICE
)

print("=" * 45)
print(f"  Final Validation Metrics  (top-{TOP_K})")
print("=" * 45)
print(f"  Loss       : {loss_final:.4f}")
print(f"  HR@{TOP_K}   : {hr_final:.4f}")
print(f"  NDCG@{TOP_K} : {ndcg_final:.4f}")
print(f"  Composite  : {comp_final:.4f}  (ALPHA={ALPHA} × nDCG + {1-ALPHA} × HR)")
print("=" * 45)

best_comp   = 0.0
patience_ct = 0
history_log = {
    "epoch"     : [],
    "train_loss": [], "train_hr": [], "train_ndcg": [], "train_comp": [],
    "val_loss"  : [], "val_hr"  : [], "val_ndcg"  : [], "val_comp"  : [],
}

# Header
print(f"{'Epoch':>5}  {'LR':>8}  "
      f"{'TrLoss':>7}  {'TrHR':>6}  {'TrNDCG':>7}  {'TrComp':>7}  "
      f"{'VaLoss':>7}  {'VaHR':>6}  {'VaNDCG':>7}  {'VaComp':>7}")
print("-" * 90)




model2 = GRU4Rec(
    num_items   = NUM_ITEMS,
    num_genres  = NUM_GENRES,
    hidden_size = D_MODEL,
    num_layers  = 2,
    dropout     = DROPOUT
).to(DEVICE)

user_id_list = [uid for uid, seq in user_sequences.items() if len(seq) >= 3]

dataset2 = SASRecDataset(train_items, train_pos, train_genres, train_ratings, user_id_list, user_histories, NUM_ITEMS)
loader2  = DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                     num_workers=0, pin_memory=(DEVICE.type == "cuda"))

optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR, betas=(0.9, 0.98))

# ── Scheduler: halve LR after 5 epochs without comp improvement ──────────────
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer2, mode="max", factor=0.5, patience=5
)


for epoch in range(1, EPOCHS + 1):

    model2.train()
    running_loss = 0.0

    for seq, pos, neg, genres, ratings in loader2:

        seq, pos, neg = seq.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
        genres, ratings = genres.to(DEVICE), ratings.to(DEVICE)
        # genres, ratings = None, None
        optimizer.zero_grad()
        ratings = ratings / 5.0
        B, L = seq.shape

        # Forward
        h = model2(seq, genres, ratings)              # (B, L, H)
        h_n = F.normalize(h, dim=-1)

        pos_emb = F.normalize(model2.item_emb(pos), dim=-1)
        neg_emb = F.normalize(model2.item_emb(neg), dim=-1)

        # Cosine scores
        pos_scores = (h_n * pos_emb).sum(-1, keepdim=True) / TEMPERATURE
        neg_scores = torch.einsum('bld,blnd->bln', h_n, neg_emb) / TEMPERATURE

        logits  = torch.cat([pos_scores, neg_scores], dim=-1)
        targets = torch.zeros(B, L, dtype=torch.long, device=DEVICE)
        mask    = (pos != 0)

        loss = F.cross_entropy(logits[mask], targets[mask])

        loss.backward()
        nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        optimizer2.step()

        running_loss += loss.item()

    train_ce   = running_loss / len(loader)
    current_lr = optimizer2.param_groups[0]["lr"]

    # ───────── Evaluate ─────────
    tr_hr, tr_ndcg, tr_loss, tr_comp = evaluate_model(
        model2, train_items, train_targets,
        train_genres, train_ratings,
        user_id_list, user_histories,
        NUM_ITEMS, TOP_K, DEVICE
    )

    va_hr, va_ndcg, va_loss, va_comp = evaluate_model(
        model2, val_items, val_targets,
        val_genres, val_ratings,
        user_id_list, user_histories,
        NUM_ITEMS, TOP_K, DEVICE
    )

    scheduler2.step(va_comp)

    is_best  = va_comp > best_comp
    ckpt_msg = "  best" if is_best else ""

    print(f"{epoch:>5}  {current_lr:>8.6f}  "
          f"{train_ce:>7.4f}  {tr_hr:>6.4f}  {tr_ndcg:>7.4f}  {tr_comp:>7.4f}  "
          f"{va_loss:>7.4f}  {va_hr:>6.4f}  {va_ndcg:>7.4f}  {va_comp:>7.4f}"
          f"{ckpt_msg}")

    if is_best:
        best_comp   = va_comp
        patience_ct = 0
        torch.save(model2.state_dict(), "best_gru4rec.pt")
    else:
        patience_ct += 1
        if epoch > WARMUP_EPOCHS and patience_ct >= ES_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}  "
                  f"(best Comp = {best_comp:.4f})")
            break

print(f"\nTraining complete. Best Val Composite: {best_comp:.4f}")

model2.load_state_dict(torch.load("best_gru4rec.pt", map_location=DEVICE))

hr_final, ndcg_final, loss_final, comp_final = evaluate_model(   # ← 4 values
    model2, val_items, val_targets, val_genres, val_ratings, user_id_list,
    user_histories, NUM_ITEMS, TOP_K, DEVICE
)

print("=" * 45)
print(f"  Final Validation Metrics  (top-{TOP_K})")
print("=" * 45)
print(f"  Loss       : {loss_final:.4f}")
print(f"  HR@{TOP_K}   : {hr_final:.4f}")
print(f"  NDCG@{TOP_K} : {ndcg_final:.4f}")
print(f"  Composite  : {comp_final:.4f}  (ALPHA={ALPHA} × nDCG + {1-ALPHA} × HR)")
print("=" * 45)