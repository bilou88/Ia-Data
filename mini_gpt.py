import os, json, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 0) Chargement du corpus
# =========================
def load_corpus(path="corpus.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        print(f"[OK] Corpus chargé: {path} | {len(txt):,} caractères")
        return txt
    # fallback si aucun fichier trouvé
    txt = (
        "bonjour je suis une intelligence artificielle\n"
        "je parle avec toi\n"
        "tu peux me poser des questions\n"
    )
    print("[INFO] Aucun corpus.txt trouvé, utilisation d'un petit texte de démo.")
    return txt

text = load_corpus("corpus.txt")

# Vocabulaire (caractères)
chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

def encode(s):      # texte -> ids (strict)
    return [stoi[c] for c in s if c in stoi]
def encode_safe(s): # texte -> ids (tolérant)
    if not s:
        return [0]
    return [stoi.get(c, 0) for c in s]
def decode(ids):    # ids -> texte
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
print(f"[STATS] Vocab size: {len(chars)} | Longueur data: {len(data):,}")

# =========================
# 1) Paramètres
# =========================
# Contexte: <=64 (et <= len(data)-1)
block_size = 128
batch_size  = 64
max_iters   = 5000
eval_interval = 400
eval_iters  = 100
learning_rate = 3e-3

embed_dim = 256
n_heads   = 8
n_layers  = 12
dropout   = 0.1
vocab_size = len(chars)

# Device (CPU / CUDA / MPS pour Mac Apple Silicon)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"[DEVICE] {device}")

# =========================
# 2) Split train / val
# =========================
n = len(data)
split = int(n * 0.9) if n > 1000 else max(1, int(n * 0.8))
train_data = data[:split]
val_data   = data[split:] if split < n else data[:1]  # évite vide

def get_batch(split_name):
    d = train_data if split_name == "train" else val_data
    # sécurité: si val trop petite, on picore dans train
    if len(d) <= block_size + 1:
        d = train_data
    ix = torch.randint(0, len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split_name in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split_name)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            losses.append(loss.item())
        out[split_name] = sum(losses) / len(losses)
    model.train()
    return out

# =========================
# 3) Modèle (Mini GPT)
# =========================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)               # (B,T,hs)
        q = self.query(x)             # (B,T,hs)
        wei = q @ k.transpose(-2, -1) # (B,T,T)
        wei = wei / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        v = self.value(x)             # (B,T,hs)
        out = wei @ v                 # (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(num_heads * head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        head_size = embed_dim // n_heads
        self.sa   = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(embed_dim)
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(n_heads) for _ in range(n_layers)])
        self.ln_f   = nn.LayerNorm(embed_dim)
        self.head   = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        # Sécurité (au cas où)
        if T > block_size:
            idx = idx[:, -block_size:]
            T = block_size
        tok_emb = self.token_embedding(idx)                               # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                             # (B,T,vocab)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)[:, -1, :]  # (B,vocab)
            logits = logits / max(1e-6, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                thresh = v[:, [-1]]
                logits = torch.where(logits < thresh, torch.tensor(-1e10, device=logits.device), logits)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =========================
# 4) Entraînement
# =========================
torch.manual_seed(42)
model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("[TRAIN] Démarrage…")
for it in range(max_iters):
    if it % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {it:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")

    xb, yb = get_batch("train")
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Sauvegarde (facultatif)
torch.save({"model_state": model.state_dict(),
            "config": {
                "embed_dim": embed_dim, "n_heads": n_heads, "n_layers": n_layers,
                "block_size": block_size, "vocab_size": vocab_size
            },
            "vocab": {"stoi": stoi, "itos": itos}
            }, "mini_gpt.pt")
print("[SAVE] Modèle sauvegardé -> mini_gpt.pt")

# =========================
# 5) Génération de texte
# =========================
def sample(prompt, max_new_tokens=400, temperature=0.8, top_k=50):
    start_ids = torch.tensor([encode_safe(prompt)], dtype=torch.long, device=device)
    out = model.generate(start_ids, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return decode(out[0].tolist())

print("\n--- ÉCHANTILLONS ---")
for temp in (0.7, 1.0, 1.2):
    print(f"\n[TEMP={temp}]")
    print(sample(prompt="je ", max_new_tokens=300, temperature=temp, top_k=50))