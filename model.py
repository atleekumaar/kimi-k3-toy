# =========================================================================
# Kimi K3 - minimal implementation
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# Config (toy scale)
# -------------------------------------------------------------------------
class K3Config:
    d_model = 64
    n_heads = 2
    d_head = 32                 # n_heads * d_head == d_model
    n_blocks = 1                 # each block = 3x KDA + 1x Gated MLA (s2.1)
    n_shared_experts = 2          # s2.3, Ns=2 in the paper
    n_routed_experts = 4
    top_k_experts = 2
    d_expert_latent = 32          # routed-expert latent width l (s2.3)
    gmin = -5.0                   # lower-bounded decay floor (s2.1.1, Eq.5)
    conv_kernel = 4
    max_seq_len = 32
    vocab_size = None             # set from dataset below

# -------------------------------------------------------------------------
# Building blocks
# -------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

class ShortConv(nn.Module):
    """Causal depthwise short convolution used ahead of q/k/v in KDA (s2.1.1, Eq.2)."""
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=0)
    def forward(self, x):  # x: [B,T,D]
        B, T, D = x.shape
        x = x.transpose(1, 2)                       # B,D,T
        x = F.pad(x, (self.kernel_size - 1, 0))      # left-pad only -> causal
        x = self.conv(x)
        return x.transpose(1, 2)                     # B,T,D

class SwiGLUExpert(nn.Module):
    """Full-width FFN used for MoE shared experts."""
    def __init__(self, d_in, d_out, hidden_mult=2):
        super().__init__()
        h = d_in * hidden_mult
        self.Wg = nn.Linear(d_in, h, bias=False)
        self.Wu = nn.Linear(d_in, h, bias=False)
        self.Wd = nn.Linear(h, d_out, bias=False)
    def forward(self, x):
        return self.Wd(F.silu(self.Wg(x)) * self.Wu(x))

class SiTUExpert(nn.Module):
    """SiTU-GLU FFN (§2.3.2, Eq.12) used for MoE routed experts (compact latent width)."""
    def __init__(self, d, hidden_mult=2, beta1=4.0, beta2=25.0):
        super().__init__()
        h = d * hidden_mult
        self.Wg = nn.Linear(d, h, bias=False)
        self.Wu = nn.Linear(d, h, bias=False)
        self.Wd = nn.Linear(h, d, bias=False)
        self.beta1, self.beta2 = beta1, beta2
    def forward(self, x):
        g, u = self.Wg(x), self.Wu(x)
        gate = self.beta1 * torch.tanh(g / self.beta1) * torch.sigmoid(g)
        up   = self.beta2 * torch.tanh(u / self.beta2)
        return self.Wd(gate * up)

# -------------------------------------------------------------------------
# Kimi Delta Attention (2.1.1) — sequential recurrent form
# -------------------------------------------------------------------------
class KDA(nn.Module):
    def __init__(self, d_model, n_heads, d_head, gmin=-5.0, conv_kernel=4):
        super().__init__()
        self.n_heads, self.d_head, self.gmin = n_heads, d_head, gmin
        inner = n_heads * d_head

        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.q_conv = ShortConv(inner, conv_kernel)
        self.k_conv = ShortConv(inner, conv_kernel)
        self.v_conv = ShortConv(inner, conv_kernel)

        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)   # scalar-per-head β_t

        r = max(8, inner // 4)                                     # low-rank decay logits (Eq.2)
        self.alpha_down = nn.Linear(d_model, r, bias=False)
        self.alpha_up = nn.Linear(r, inner, bias=False)
        self.alpha_bias = nn.Parameter(torch.zeros(inner))
        self.A_log_scale = nn.Parameter(torch.zeros(n_heads))      # per-head A_h, init 0 (Eq.5)

        self.out_norm = RMSNorm(d_head)                            # head-wise RMSNorm (Eq.6)
        self.gate_proj = nn.Linear(d_model, inner, bias=True)      # full-rank output gate
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = F.silu(self.q_conv(self.q_proj(x))).view(B, T, H, Dh)
        k = F.silu(self.k_conv(self.k_proj(x))).view(B, T, H, Dh)
        v = F.silu(self.v_conv(self.v_proj(x))).view(B, T, H, Dh)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        beta = torch.sigmoid(self.beta_proj(x))                    # B,T,H

        z = (self.alpha_up(self.alpha_down(x)) + self.alpha_bias).view(B, T, H, Dh)
        A = self.A_log_scale.view(1, 1, H, 1)
        g = self.gmin * torch.sigmoid(torch.exp(A) * z)            # Eq.5
        alpha = torch.exp(g)                                       # channel-wise decay, (e^gmin, 1)

        S = x.new_zeros(B, H, Dh, Dh)                               # recurrent state, dk x dv
        outs = []
        for t in range(T):                                          # Eq.1, unrolled in time
            k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]
            a_t, b_t = alpha[:, t], beta[:, t]

            S = a_t.unsqueeze(-1) * S                                # Diag(alpha_t) S_{t-1}
            kv_proj = torch.einsum('bhd,bhde->bhe', k_t, S)          # k_t^T S
            S = S - b_t.view(B, H, 1, 1) * k_t.unsqueeze(-1) * kv_proj.unsqueeze(-2)
            S = S + b_t.view(B, H, 1, 1) * k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            o_t = torch.einsum('bhd,bhde->bhe', q_t, S)              # S_t^T q_t
            outs.append(o_t)

        o = self.out_norm(torch.stack(outs, dim=1)).reshape(B, T, H * Dh)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.out_proj(gate * o)                               # Eq.6

# -------------------------------------------------------------------------
# Gated MLA (2.1.2) — NoPE causal attention + output gate
# -------------------------------------------------------------------------
class GatedMLA(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        inner = n_heads * d_head
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.gate_proj = nn.Linear(d_model, inner, bias=True)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.scale = d_head ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        attn = torch.einsum('bhtd,bhsd->bhts', q, k) * self.scale     # no positional encoding
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf')).softmax(dim=-1)
        o = torch.einsum('bhts,bhsd->bhtd', attn, v).transpose(1, 2).reshape(B, T, H * Dh)

        gate = torch.sigmoid(self.gate_proj(x))                        # Eq.7
        return self.out_proj(gate * o)

# -------------------------------------------------------------------------
# Stable LatentMoE (2.3) — shared + routed experts, dense-compute router
# -------------------------------------------------------------------------
class StableLatentMoE(nn.Module):
    def __init__(self, d_model, d_latent, n_shared, n_routed, top_k, hidden_mult=2):
        super().__init__()
        self.n_routed, self.top_k = n_routed, top_k
        self.shared_experts = nn.ModuleList(
            [SwiGLUExpert(d_model, d_model, hidden_mult) for _ in range(n_shared)])
        self.down_proj = nn.Linear(d_model, d_latent, bias=False)      # W_down
        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.routed_experts = nn.ModuleList(
            [SiTUExpert(d_latent, hidden_mult) for _ in range(n_routed)])
        self.pre_up_norm = RMSNorm(d_latent)                            # 2.3.1
        self.up_proj = nn.Linear(d_latent, d_model, bias=False)         # W_up

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)

        shared_out = sum(e(xf) for e in self.shared_experts)            # Eq.11, shared term

        scores = torch.sigmoid(self.router(xf))                         # Eq.13 router
        topk_val, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_weight = topk_val / topk_val.sum(-1, keepdim=True).clamp_min(1e-9)
        full_weight = torch.zeros_like(scores).scatter(-1, topk_idx, topk_weight)  # dense, mostly 0

        z = self.down_proj(xf)                                          # routed latent (Eq.11)
        u = z.new_zeros(z.shape)
        for e_idx in range(self.n_routed):                               # dense-compute simplification
            u = u + full_weight[:, e_idx:e_idx + 1] * self.routed_experts[e_idx](z)

        routed_out = self.up_proj(self.pre_up_norm(u))                   # Eq.11
        return (shared_out + routed_out).reshape(B, T, D)

# -------------------------------------------------------------------------
# Attention Residuals (2.2) — Full form, Eq.8-9
# -------------------------------------------------------------------------
class AttnRes(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.pseudo_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(d_model) * 0.02) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)

    def compute_h(self, layer_idx, history):
        # history: list of [B,T,D] tensors — embedding + every previous layer's output
        w = self.pseudo_queries[layer_idx]
        V = torch.stack(history, dim=2)                    # B,T,N,D
        K = self.norm(V)                                    # phi(q,k) kernel numerator (Eq.9)
        scores = torch.einsum('btnd,d->btn', K, w)
        attn = scores.softmax(dim=-1)
        return torch.einsum('btn,btnd->btd', attn, V)        # h_l, Eq.9

# -------------------------------------------------------------------------
# Assemble the model
# -------------------------------------------------------------------------
class SubBlock(nn.Module):
    def __init__(self, cfg, layer_type):
        super().__init__()
        self.attn = (KDA(cfg.d_model, cfg.n_heads, cfg.d_head, cfg.gmin, cfg.conv_kernel)
                     if layer_type == 'kda' else GatedMLA(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.moe = StableLatentMoE(cfg.d_model, cfg.d_expert_latent,
                                    cfg.n_shared_experts, cfg.n_routed_experts, cfg.top_k_experts)

class KimiK3Mini(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        layer_types = (['kda', 'kda', 'kda', 'mla']) * cfg.n_blocks   # 2.1, 3:1 KDA:MLA ratio
        self.sub_blocks = nn.ModuleList([SubBlock(cfg, t) for t in layer_types])
        self.attnres = AttnRes(cfg.d_model, len(self.sub_blocks))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        history = [x]                                       # v_0 = h_1 = embedding (§2.2)
        for l, block in enumerate(self.sub_blocks):
            h_l = self.attnres.compute_h(l, history)          # AttnRes replaces the plain residual
            y = h_l + block.attn(h_l)
            y = y + block.moe(y)
            history.append(y)
        return self.lm_head(self.final_norm(history[-1]))

# =========================================================================
# Verification: toy synthetic dataset (periodic char sequence), train, generate
# =========================================================================
pattern = "0123456789ABCDEF"      # synthetic, no copyright concerns
text = pattern * 200
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

cfg = K3Config()
cfg.vocab_size = len(chars)

model = KimiK3Mini(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model built. Trainable parameters: {n_params:,}")

# --- sanity forward+backward pass before training ---
xb0 = data[:cfg.max_seq_len].unsqueeze(0).to(device)
yb0 = data[1:cfg.max_seq_len + 1].unsqueeze(0).to(device)
logits0 = model(xb0)
print(f"Sanity check — logits shape: {tuple(logits0.shape)} (expect [1, {cfg.max_seq_len}, {cfg.vocab_size}])")
loss0 = F.cross_entropy(logits0.view(-1, cfg.vocab_size), yb0.view(-1))
loss0.backward()
n_nan_grads = sum(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None)
print(f"Sanity check — initial loss: {loss0.item():.4f}, NaN grads: {n_nan_grads}")
model.zero_grad()

# --- training loop ---
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
n_steps, batch_size = 300, 16
print("\nTraining on synthetic periodic sequence (verifies grads flow end-to-end)...")
for step in range(n_steps):
    xb, yb = get_batch(data, cfg.max_seq_len, batch_size, device)
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), yb.view(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0 or step == n_steps - 1:
        print(f"  step {step:4d} | loss {loss.item():.4f}")

# --- generation ---
@torch.no_grad()
def generate(model, start_idx, n_new):
    model.eval()
    idx = start_idx.clone()
    for _ in range(n_new):
        logits = model(idx)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return idx

start = data[:8].unsqueeze(0).to(device)
gen = generate(model, start, 48)[0].tolist()
print("\nGenerated (should show visible periodicity if training worked):")
print(''.join(itos[i] for i in gen))# =========================================================================
# Kimi K3 - minimal implementation
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# Config (toy scale)
# -------------------------------------------------------------------------
class K3Config:
    d_model = 64
    n_heads = 2
    d_head = 32                 # n_heads * d_head == d_model
    n_blocks = 1                 # each block = 3x KDA + 1x Gated MLA (s2.1)
    n_shared_experts = 2          # s2.3, Ns=2 in the paper
    n_routed_experts = 4
    top_k_experts = 2
    d_expert_latent = 32          # routed-expert latent width l (s2.3)
    gmin = -5.0                   # lower-bounded decay floor (s2.1.1, Eq.5)
    conv_kernel = 4
    max_seq_len = 32
    vocab_size = None             # set from dataset below

# -------------------------------------------------------------------------
# Building blocks
# -------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

class ShortConv(nn.Module):
    """Causal depthwise short convolution used ahead of q/k/v in KDA (s2.1.1, Eq.2)."""
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=0)
    def forward(self, x):  # x: [B,T,D]
        B, T, D = x.shape
        x = x.transpose(1, 2)                       # B,D,T
        x = F.pad(x, (self.kernel_size - 1, 0))      # left-pad only -> causal
        x = self.conv(x)
        return x.transpose(1, 2)                     # B,T,D

class SwiGLUExpert(nn.Module):
    """Full-width FFN used for MoE shared experts."""
    def __init__(self, d_in, d_out, hidden_mult=2):
        super().__init__()
        h = d_in * hidden_mult
        self.Wg = nn.Linear(d_in, h, bias=False)
        self.Wu = nn.Linear(d_in, h, bias=False)
        self.Wd = nn.Linear(h, d_out, bias=False)
    def forward(self, x):
        return self.Wd(F.silu(self.Wg(x)) * self.Wu(x))

class SiTUExpert(nn.Module):
    """SiTU-GLU FFN (§2.3.2, Eq.12) used for MoE routed experts (compact latent width)."""
    def __init__(self, d, hidden_mult=2, beta1=4.0, beta2=25.0):
        super().__init__()
        h = d * hidden_mult
        self.Wg = nn.Linear(d, h, bias=False)
        self.Wu = nn.Linear(d, h, bias=False)
        self.Wd = nn.Linear(h, d, bias=False)
        self.beta1, self.beta2 = beta1, beta2
    def forward(self, x):
        g, u = self.Wg(x), self.Wu(x)
        gate = self.beta1 * torch.tanh(g / self.beta1) * torch.sigmoid(g)
        up   = self.beta2 * torch.tanh(u / self.beta2)
        return self.Wd(gate * up)

# -------------------------------------------------------------------------
# Kimi Delta Attention (2.1.1) — sequential recurrent form
# -------------------------------------------------------------------------
class KDA(nn.Module):
    def __init__(self, d_model, n_heads, d_head, gmin=-5.0, conv_kernel=4):
        super().__init__()
        self.n_heads, self.d_head, self.gmin = n_heads, d_head, gmin
        inner = n_heads * d_head

        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.q_conv = ShortConv(inner, conv_kernel)
        self.k_conv = ShortConv(inner, conv_kernel)
        self.v_conv = ShortConv(inner, conv_kernel)

        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)   # scalar-per-head β_t

        r = max(8, inner // 4)                                     # low-rank decay logits (Eq.2)
        self.alpha_down = nn.Linear(d_model, r, bias=False)
        self.alpha_up = nn.Linear(r, inner, bias=False)
        self.alpha_bias = nn.Parameter(torch.zeros(inner))
        self.A_log_scale = nn.Parameter(torch.zeros(n_heads))      # per-head A_h, init 0 (Eq.5)

        self.out_norm = RMSNorm(d_head)                            # head-wise RMSNorm (Eq.6)
        self.gate_proj = nn.Linear(d_model, inner, bias=True)      # full-rank output gate
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = F.silu(self.q_conv(self.q_proj(x))).view(B, T, H, Dh)
        k = F.silu(self.k_conv(self.k_proj(x))).view(B, T, H, Dh)
        v = F.silu(self.v_conv(self.v_proj(x))).view(B, T, H, Dh)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        beta = torch.sigmoid(self.beta_proj(x))                    # B,T,H

        z = (self.alpha_up(self.alpha_down(x)) + self.alpha_bias).view(B, T, H, Dh)
        A = self.A_log_scale.view(1, 1, H, 1)
        g = self.gmin * torch.sigmoid(torch.exp(A) * z)            # Eq.5
        alpha = torch.exp(g)                                       # channel-wise decay, (e^gmin, 1)

        S = x.new_zeros(B, H, Dh, Dh)                               # recurrent state, dk x dv
        outs = []
        for t in range(T):                                          # Eq.1, unrolled in time
            k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]
            a_t, b_t = alpha[:, t], beta[:, t]

            S = a_t.unsqueeze(-1) * S                                # Diag(alpha_t) S_{t-1}
            kv_proj = torch.einsum('bhd,bhde->bhe', k_t, S)          # k_t^T S
            S = S - b_t.view(B, H, 1, 1) * k_t.unsqueeze(-1) * kv_proj.unsqueeze(-2)
            S = S + b_t.view(B, H, 1, 1) * k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            o_t = torch.einsum('bhd,bhde->bhe', q_t, S)              # S_t^T q_t
            outs.append(o_t)

        o = self.out_norm(torch.stack(outs, dim=1)).reshape(B, T, H * Dh)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.out_proj(gate * o)                               # Eq.6

# -------------------------------------------------------------------------
# Gated MLA (2.1.2) — NoPE causal attention + output gate
# -------------------------------------------------------------------------
class GatedMLA(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        inner = n_heads * d_head
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.gate_proj = nn.Linear(d_model, inner, bias=True)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.scale = d_head ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        attn = torch.einsum('bhtd,bhsd->bhts', q, k) * self.scale     # no positional encoding
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf')).softmax(dim=-1)
        o = torch.einsum('bhts,bhsd->bhtd', attn, v).transpose(1, 2).reshape(B, T, H * Dh)

        gate = torch.sigmoid(self.gate_proj(x))                        # Eq.7
        return self.out_proj(gate * o)

# -------------------------------------------------------------------------
# Stable LatentMoE (2.3) — shared + routed experts, dense-compute router
# -------------------------------------------------------------------------
class StableLatentMoE(nn.Module):
    def __init__(self, d_model, d_latent, n_shared, n_routed, top_k, hidden_mult=2):
        super().__init__()
        self.n_routed, self.top_k = n_routed, top_k
        self.shared_experts = nn.ModuleList(
            [SwiGLUExpert(d_model, d_model, hidden_mult) for _ in range(n_shared)])
        self.down_proj = nn.Linear(d_model, d_latent, bias=False)      # W_down
        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.routed_experts = nn.ModuleList(
            [SiTUExpert(d_latent, hidden_mult) for _ in range(n_routed)])
        self.pre_up_norm = RMSNorm(d_latent)                            # 2.3.1
        self.up_proj = nn.Linear(d_latent, d_model, bias=False)         # W_up

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)

        shared_out = sum(e(xf) for e in self.shared_experts)            # Eq.11, shared term

        scores = torch.sigmoid(self.router(xf))                         # Eq.13 router
        topk_val, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_weight = topk_val / topk_val.sum(-1, keepdim=True).clamp_min(1e-9)
        full_weight = torch.zeros_like(scores).scatter(-1, topk_idx, topk_weight)  # dense, mostly 0

        z = self.down_proj(xf)                                          # routed latent (Eq.11)
        u = z.new_zeros(z.shape)
        for e_idx in range(self.n_routed):                               # dense-compute simplification
            u = u + full_weight[:, e_idx:e_idx + 1] * self.routed_experts[e_idx](z)

        routed_out = self.up_proj(self.pre_up_norm(u))                   # Eq.11
        return (shared_out + routed_out).reshape(B, T, D)

# -------------------------------------------------------------------------
# Attention Residuals (2.2) — Full form, Eq.8-9
# -------------------------------------------------------------------------
class AttnRes(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.pseudo_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(d_model) * 0.02) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)

    def compute_h(self, layer_idx, history):
        # history: list of [B,T,D] tensors — embedding + every previous layer's output
        w = self.pseudo_queries[layer_idx]
        V = torch.stack(history, dim=2)                    # B,T,N,D
        K = self.norm(V)                                    # phi(q,k) kernel numerator (Eq.9)
        scores = torch.einsum('btnd,d->btn', K, w)
        attn = scores.softmax(dim=-1)
        return torch.einsum('btn,btnd->btd', attn, V)        # h_l, Eq.9

# -------------------------------------------------------------------------
# Assemble the model
# -------------------------------------------------------------------------
class SubBlock(nn.Module):
    def __init__(self, cfg, layer_type):
        super().__init__()
        self.attn = (KDA(cfg.d_model, cfg.n_heads, cfg.d_head, cfg.gmin, cfg.conv_kernel)
                     if layer_type == 'kda' else GatedMLA(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.moe = StableLatentMoE(cfg.d_model, cfg.d_expert_latent,
                                    cfg.n_shared_experts, cfg.n_routed_experts, cfg.top_k_experts)

class KimiK3Mini(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        layer_types = (['kda', 'kda', 'kda', 'mla']) * cfg.n_blocks   # 2.1, 3:1 KDA:MLA ratio
        self.sub_blocks = nn.ModuleList([SubBlock(cfg, t) for t in layer_types])
        self.attnres = AttnRes(cfg.d_model, len(self.sub_blocks))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        history = [x]                                       # v_0 = h_1 = embedding (§2.2)
        for l, block in enumerate(self.sub_blocks):
            h_l = self.attnres.compute_h(l, history)          # AttnRes replaces the plain residual
            y = h_l + block.attn(h_l)
            y = y + block.moe(y)
            history.append(y)
        return self.lm_head(self.final_norm(history[-1]))

# =========================================================================
# Verification: toy synthetic dataset (periodic char sequence), train, generate
# =========================================================================
pattern = "0123456789ABCDEF"      # synthetic, no copyright concerns
text = pattern * 200
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

cfg = K3Config()
cfg.vocab_size = len(chars)

model = KimiK3Mini(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model built. Trainable parameters: {n_params:,}")

# --- sanity forward+backward pass before training ---
xb0 = data[:cfg.max_seq_len].unsqueeze(0).to(device)
yb0 = data[1:cfg.max_seq_len + 1].unsqueeze(0).to(device)
logits0 = model(xb0)
print(f"Sanity check — logits shape: {tuple(logits0.shape)} (expect [1, {cfg.max_seq_len}, {cfg.vocab_size}])")
loss0 = F.cross_entropy(logits0.view(-1, cfg.vocab_size), yb0.view(-1))
loss0.backward()
n_nan_grads = sum(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None)
print(f"Sanity check — initial loss: {loss0.item():.4f}, NaN grads: {n_nan_grads}")
model.zero_grad()

# --- training loop ---
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
n_steps, batch_size = 300, 16
print("\nTraining on synthetic periodic sequence (verifies grads flow end-to-end)...")
for step in range(n_steps):
    xb, yb = get_batch(data, cfg.max_seq_len, batch_size, device)
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), yb.view(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0 or step == n_steps - 1:
        print(f"  step {step:4d} | loss {loss.item():.4f}")

# --- generation ---
@torch.no_grad()
def generate(model, start_idx, n_new):
    model.eval()
    idx = start_idx.clone()
    for _ in range(n_new):
        logits = model(idx)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return idx

start = data[:8].unsqueeze(0).to(device)
gen = generate(model, start, 48)[0].tolist()
print("\nGenerated (should show visible periodicity if training worked):")
print(''.join(itos[i] for i in gen)) 
