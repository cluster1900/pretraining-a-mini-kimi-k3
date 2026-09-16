"""Optional DeepSeek-Coder-style dense decoder for comparison experiments.

This is a clean-room implementation of the public design pattern: RMSNorm,
RoPE, causal attention and SwiGLU. It is intentionally separate from Mini K3's
KDA/MLA/MoE backbone so checkpoints cannot be mixed accidentally.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekCoderConfig:
    def __init__(self, vocab_size=163840, hidden_size=2048, layers=24,
                 heads=16, intermediate_size=5504, max_seq_len=4096):
        self.vocab_size, self.hidden_size, self.layers = vocab_size, hidden_size, layers
        self.heads, self.intermediate_size, self.max_seq_len = heads, intermediate_size, max_seq_len
        self.attention_window = 4096

class _Block(nn.Module):
    def __init__(self, c):
        super().__init__(); self.n1=nn.RMSNorm(c.hidden_size); self.n2=nn.RMSNorm(c.hidden_size)
        self.qkv=nn.Linear(c.hidden_size, 3*c.hidden_size, bias=False); self.o=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
        self.up=nn.Linear(c.hidden_size, 2*c.intermediate_size, bias=False); self.down=nn.Linear(c.intermediate_size,c.hidden_size,bias=False)
        self.h=c.hidden_size; self.heads=c.heads
    def forward(self,x):
        b,l,_=x.shape; q,k,v=self.qkv(self.n1(x)).chunk(3,-1); d=self.h//self.heads
        q=q.view(b,l,self.heads,d).transpose(1,2); k=k.view(b,l,self.heads,d).transpose(1,2); v=v.view(b,l,self.heads,d).transpose(1,2)
        pos=torch.arange(l,device=x.device); inv=1/(10000**(torch.arange(0,d,2,device=x.device).float()/d)); a=pos[:,None]*inv[None,:]
        cos=torch.repeat_interleave(a.cos(),2,-1)[None,None]; sin=torch.repeat_interleave(a.sin(),2,-1)[None,None]
        def rot(z): return z*cos + torch.cat((-z[...,d//2:],z[...,:d//2]),-1)*sin
        y=F.scaled_dot_product_attention(rot(q),rot(k),v,is_causal=True); x=x+self.o(y.transpose(1,2).reshape(b,l,self.h))
        u,g=self.up(self.n2(x)).chunk(2,-1); return x+self.down(F.silu(g)*u)

class DeepSeekCoderForCausalLM(nn.Module):
    def __init__(self, config=None):
        super().__init__(); c=config or DeepSeekCoderConfig(); self.config=c
        self.embed=nn.Embedding(c.vocab_size,c.hidden_size); self.blocks=nn.ModuleList([_Block(c) for _ in range(c.layers)]); self.norm=nn.RMSNorm(c.hidden_size); self.lm_head=nn.Linear(c.hidden_size,c.vocab_size,bias=False); self.lm_head.weight=self.embed.weight
    def forward(self,input_ids,labels=None):
        x=self.embed(input_ids)
        for b in self.blocks: x=b(x)
        logits=self.lm_head(self.norm(x)); loss=None
        if labels is not None: loss=F.cross_entropy(logits[...,:-1,:].reshape(-1,logits.size(-1)),labels[...,1:].reshape(-1),ignore_index=-100)
        return {"logits":logits,"loss":loss}
