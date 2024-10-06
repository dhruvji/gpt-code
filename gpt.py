import torch
import torch.nn as nn
import math

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "embedding dimension must be divisible by number of heads"
        
        # Linear layer for query, key, and value
        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        
        self.d_model = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = self.d_model // self.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Causal mask for self-attention
        self.mask = torch.tril(torch.ones(1, 1, cfg.n_ctx, cfg.n_ctx))

    def forward(self, x):
        bs, seq_len, _ = x.shape
        
        # Project input to query, key, and value
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # Reshape for multi-head attention and transpose for scaled dot-product attention
        q = q.view(bs, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with masking
        attn_weights = (q @ k.transpose(-1, -2)) * self.scale
        attn_weights = attn_weights.masked_fill(self.mask[..., :seq_len, :seq_len] == 0, -float('inf'))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        # Compute attention output
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        return self.out_proj(attn_output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class GPT2Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MaskedMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForwardNetwork(cfg)

    def forward(self, x):
        # Apply masked multi-head attention
        attn_output = self.attn(self.ln1(x))
        x = x + attn_output
        
        # Apply feed-forward network
        ffn_output = self.ffn(self.ln2(x))
        x = x + ffn_output
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.n_ctx, cfg.n_embd)
        
        # Stack GPT2 blocks
        self.blocks = nn.ModuleList([GPT2Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, x):
        bs, seq_len = x.shape
        assert seq_len <= self.cfg.n_ctx, "Input sequence length exceeds model context size"
        
        # Generate token and position embeddings
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(pos)
        
        # Pass through GPT blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer normalization
        x = self.ln_f(x)
        
        # Only use the last token's logits for language modeling
        logits = self.lm_head(x[:, -1:, :])
        return logits

    @torch.no_grad()
    def generate(self, token_idx, max_new_tokens=200, temperature=1.0, top_k=50, do_sample=True):
        for _ in range(max_new_tokens):
            # Truncate input to the model's maximum context size
            token_idx = token_idx if token_idx.size(1) <= self.cfg.n_ctx else token_idx[:, -self.cfg.n_ctx:]
            logits = self(token_idx)

            # Extract the last token's logits and apply temperature scaling
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('inf')

            # Sample or take the argmax of the probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) if do_sample else torch.argmax(probs, dim=-1, keepdim=True)
            
            # Append the generated token to the sequence
            token_idx = torch.cat([token_idx, next_token], dim=1)

        return token_idx
