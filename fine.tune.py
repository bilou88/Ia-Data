import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Charger le modèle existant
checkpoint = torch.load("mini_gpt.pt", map_location="cpu")
config = checkpoint["config"]
stoi = checkpoint["vocab"]["stoi"]
itos = checkpoint["vocab"]["itos"]

vocab_size = config["vocab_size"]
embed_dim  = config["embed_dim"]
n_heads    = config["n_heads"]
n_layers   = config["n_layers"]
block_size = config["block_size"]

# ======= MiniGPT (même archi que l'entraînement) =======
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) / (C**0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
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
        if T > block_size:
            idx = idx[:, -block_size:]
            T = block_size
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

# Reconstruire et charger poids
model = MiniGPT()
model.load_state_dict(checkpoint["model_state"])
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# ======= Charger le fichier QA =======
with open("qa.txt", "r", encoding="utf-8") as f:
    qa_text = f.read()

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(qa_text), dtype=torch.long)

# ======= Fine-tuning =======
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("[Fine-tuning sur qa.txt]")
for step in range(5000):  # nombre d’itérations ajustable
    ix = torch.randint(0, len(data) - block_size - 1, (1,))
    x = data[ix:ix+block_size].unsqueeze(0).to(device)
    y = data[ix+1:ix+1+block_size].unsqueeze(0).to(device)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss {loss.item():.4f}")

# Sauvegarde nouvelle version
torch.save({
    "model_state": model.state_dict(),
    "config": config,
    "vocab": {"stoi": stoi, "itos": itos}
}, "mini_gpt_qa.pt")
print("[OK] Fine-tuning terminé. Nouveau modèle -> mini_gpt_qa.pt")
