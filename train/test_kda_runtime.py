"""KDA state precision, gradient connectivity and chunk equivalence regression."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train.config import MiniK3Config
from train.models.kda import KimiDeltaAttention


def main():
    torch.manual_seed(5)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    c=MiniK3Config(hidden_size=32,num_kda_heads=2,head_dim=16)
    m=KimiDeltaAttention(c).to(device)
    x=torch.randn(2,17,32,device=device)
    full,st,conv=m(x)
    first,s,cs=m(x[:,:9]);second,s,cs=m(x[:,9:],s,cs)
    assert torch.allclose(full,torch.cat([first,second],1),atol=2e-5,rtol=2e-5)
    full.square().mean().backward()
    assert m.A_log.grad is not None and m.A_log.grad.isfinite().all() and m.A_log.grad.abs().sum()>0
    assert m.dt_bias.grad is not None and m.dt_bias.grad.isfinite().all()
    if device=='cuda':
        with torch.autocast('cuda',dtype=torch.float16):
            y,s,cs=m(x)
        assert s.dtype==torch.float32 and s.isfinite().all() and y.isfinite().all()
    print('KDA_STATE_GRADIENT_CHUNK_TEST_PASSED')

if __name__=='__main__':main()
