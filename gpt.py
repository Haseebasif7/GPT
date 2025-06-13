import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
n_embd = 384 # size of the embedding vector for each token
max_iters = 5000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
heads_count = 6 # number of attention heads
dropout = 0.2
n_layer = 6 # number of transformer blocks
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) 
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

# class LayerNorm(nn.Module):
  
#   def __init__(self, dim, eps=1e-5, momentum=0.1):
#     super().__init__()
#     self.eps = eps
#     self.momentum = momentum
#     self.dim=dim
#     # parameters (trained with backprop)
#     self.gamma = torch.ones(self.dim)
#     self.beta = torch.zeros(self.dim)

#   def forward(self, x):
#     # calculate the forward pass
#       xmean = x.mean(1, keepdim=True) 
#       xvar = x.var(1, keepdim=True) 
#       xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
#       self.out = self.gamma * xhat + self.beta
#       return self.out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj =  nn.Linear(num_heads * head_size, n_embd) # projection layer to combine heads
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, num_heads * head_size)
        out = self.dropout(self.proj(out))
        return out # (B, T, num_heads * head_size)
        
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
          nn.Linear(n_embd, 4 * n_embd),  # expand
          nn.ReLU(),                      # non-linearity
          nn.Linear(4 * n_embd, n_embd),  # project back
          nn.Dropout(dropout) 
)

    def forward(self, x):
        return self.net(x) # (B, T, C)
    
class Block(nn.Module):
    def __init__(self,head_count):
        super().__init__()
        self.sa_heads = MultiHeadAttention(num_heads=head_count, head_size=n_embd//heads_count) # 4 heads each will output (B, T, head_size)
        self.ffwd= FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self,x):
        
        # Skip Connections
        # Pre Layer Normalization
        x = x + self.sa_heads(self.ln1(x)) # (B, T, num_heads * head_size)
        x = x + self.ffwd(self.ln2(x)) 
        return x
            
class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd ) # (65, 32)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # (8, 32)
        self.sa_heads = MultiHeadAttention(num_heads=heads_count, head_size=n_embd//heads_count) # 4 heads each will output (B, T, head_size)
        self.lm_head = nn.Linear(n_embd, vocab_size) # (32, 65)
        self.blocks =  nn.Sequential(*[ Block(head_count=4) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        
    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=idx.device)) # (T, C) where C=n_embed (8,32)
        tok_emb = self.token_embedding_table(idx) # (B,T, C) where C=n_embed (32,8,32)
        x = tok_emb + pos_emb # (B,T,C) where C=n_embed (32,8,32)
        x = self.blocks(x) # (B,T,C) where C=n_embed (32,8,32)
        x = self.ln(x) 
        logits = self.lm_head(x) # (B,T,C) where C=vocab_size (32,8,65)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT()
m = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

adam_iters = int(0.7 * max_iters)  # e.g., use AdamW for 70% of training

for iter in range(max_iters):

    if iter == adam_iters:
        print(f"🔁 Switching to SGD at iteration {iter}")
        # Switch to SGD with momentum
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate * 0.1, momentum=0.9)

    # Logging
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Training step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    break

# Generate output
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
