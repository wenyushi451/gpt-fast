import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim as optim

import math
from einops import rearrange
from einops.layers.torch import Rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, n_head, dim, is_casual, dropout=0.0, scale=None) -> None:
        super().__init__()
        self.n_head = n_head
        self.dim = dim
        head_dim = dim//n_head
        # self.linear_q = nn.Linear(dim, dim, bias=False)
        # self.linear_k = nn.Linear(dim, dim, bias=False)
        # self.linear_v = nn.Linear(dim, dim, bias=False)
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, 3*dim, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, d=head_dim, h=self.n_head),
        )
        self.head_dim = head_dim
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.is_casual = is_casual
        self.dropout = dropout
        self.scale = scale
    
    def forward(self, x):
        # x: bs, seqlen, dim
        bs, seqlen, dim = x.shape
        self.scale = self.scale if self.scale is not None else self.head_dim ** -0.5
        
        # q = self.linear_q(x)  
        # k = self.linear_k(x)
        # v = self.linear_v(x)
        
        # q = q.view(bs, seqlen, self.n_head, self.head_dim)  # bs, seqlen, n_head, head_dim @ bs, seqlen, head_dim, n_head
        # v = v.view(bs, seqlen, self.n_head, self.head_dim)  # bs, seqlen, n_head, head_dim
        # k = k.view(bs, seqlen, self.n_head, self.head_dim)  # bs, seqlen, n_head, head_dim
        q, k, v = self.to_qkv(x)
        # q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))  # bs, n_head, seqlen, head_dim
        
        if self.is_casual:
            self.casual_mask = torch.ones((seqlen, seqlen), dtype=torch.bool, device=x.device).tril(diagonal=0)
            attn_bias = torch.zeros_like(self.casual_mask, dtype=q.dtype, device=x.device).masked_fill(self.casual_mask.logical_not(), float('-inf'))
        attn_weight = q @ k.transpose(-2, -1) * self.scale  # attn: bs, n_head seqlen, seqlen
        assert attn_weight.shape == (bs, self.n_head, seqlen, seqlen)
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, train=True)
        out = attn_weight @ v
        out = out.view(bs, seqlen, self.n_head * self.head_dim)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, intermediate_size, dim) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
    
    def forward(self, x):
        ret = self.w2(F.relu(self.w1(x)) + self.w3(x))
        return ret


class TransformerBlock(nn.Module):
    def __init__(self, attention: Attention, feed_forward: FeedForward) -> None:
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.attention_norm = RMSNorm(attention.dim)
    
    def forward(self, x):
        h = self.attention(self.attention_norm(x))
        return x + self.feed_forward(h)


class Transformer(nn.Module):
    def __init__(self, n_layers=2, n_head=8, head_dim=64, vocab_size=32000, max_seq_length=2048, is_casual=True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, head_dim)
        self.positional_embedding = nn.Parameter(torch.randn(max_seq_length, head_dim))
        attn = Attention(n_head, head_dim, is_casual)
        ffn = FeedForward(head_dim, head_dim)
        self.attn = nn.ModuleList(TransformerBlock(attn, ffn) for _ in range(n_layers))
        self.out_layer = nn.Linear(head_dim, vocab_size, bias=False)
        self.is_casual = is_casual
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x_embed = self.token_embedding(x)
        x = x_embed + self.positional_embedding
        for attn in self.attn:
            x = attn(x)

        return self.out_layer(x)
    
    def generate(self, x):
        # TODO: speculate, sampling, kv cache
        pass
    

if __name__ == '__main__':
    
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    
    device_id = rank % torch.cuda.device_count()
    vocab_size = 32000
    max_seq_length = 2048
    batch_size = 8
    
    t = Transformer(4, 8, 128, vocab_size, max_seq_length).to(device_id)
    ddp_model = DDP(t, device_ids=[device_id])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    optimizer.zero_grad()
    
    inp = torch.randint(0, vocab_size, (batch_size, max_seq_length)).to(device_id)
    pred = t.forward(inp)
    y = torch.zeros((batch_size, vocab_size), dtype=torch.long).to(device_id)
    target = torch.randint(0, vocab_size, (batch_size,)).to(device_id)
    y.scatter_(1, target.unsqueeze(1), 1)
    t.ce_loss(pred, y).backward()
    optimizer.step()
    # dist.barrier()
    print(f"rank {rank}: text_emb grad: {ddp_model.module.token_embedding.weight.grad[0, :10] * 1000}")
    print(f"rank {rank}: text_emb weight after optim step: {list(t.parameters())[0][0, :10]}")
    print(f"Finish running basic DDP example on rank {rank}.")
