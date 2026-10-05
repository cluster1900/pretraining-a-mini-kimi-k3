"""KDA state precision, gradient connectivity and chunk equivalence regression."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train.config import MiniK3Config
from train.models.kda import KimiDeltaAttention, bounded_log_decay, chunk_delta_rule


def _serial_delta(q, k, v, decay, beta):
    """Token recurrence with state stored as [value, key], matching the chunk solver."""
    batches, heads, length, dim = q.shape
    state = torch.zeros(batches, heads, dim, dim, device=q.device, dtype=q.dtype)
    outputs = []
    for t in range(length):
        state = state * decay[:, :, t].unsqueeze(-2)
        pred = torch.einsum("bhvd,bhd->bhv", state, k[:, :, t])
        err = v[:, :, t] - pred
        state = state + beta[:, :, t][:, :, None, None] * err.unsqueeze(-1) * k[:, :, t].unsqueeze(-2)
        outputs.append(torch.einsum("bhvd,bhd->bhv", state, q[:, :, t]))
    return torch.stack(outputs, dim=2), state


def main():
    torch.manual_seed(5)
    z = torch.randn(2, 4, 8, 16)
    a_log = torch.zeros(4)
    log_decay = bounded_log_decay(z, a_log, -5.0)
    assert torch.isfinite(log_decay).all()
    assert (log_decay > -5.0).all() and (log_decay < 0).all()
    q = torch.randn(2, 2, 11, 8, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn(2, 2, 11, 8, dtype=torch.float64), dim=-1)
    v = torch.randn(2, 2, 11, 8, dtype=torch.float64)
    decay = torch.empty(2, 2, 11, 8, dtype=torch.float64).uniform_(0.2, 0.9)
    beta = torch.empty(2, 2, 11, dtype=torch.float64).uniform_(0.1, 0.9)
    state0 = torch.zeros(2, 2, 8, 8, dtype=torch.float64)
    serial_out, serial_state = _serial_delta(q, k, v, decay, beta)
    for chunk in (1, 3, 4, 11, 64):
        chunk_out, chunk_state = chunk_delta_rule(q, k, v, decay.log(), beta, state0.clone(), chunk_size=chunk)
        assert torch.allclose(chunk_out, serial_out, atol=1e-9, rtol=1e-9), chunk
        assert torch.allclose(chunk_state, serial_state, atol=1e-9, rtol=1e-9), chunk
    # Non-zero incoming state and gradient equality.
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v, decay, beta)]
    init = torch.randn(2, 2, 8, 8, dtype=torch.float64) * 0.1
    a_out, a_state = chunk_delta_rule(leaves[0], leaves[1], leaves[2], leaves[3].log(), leaves[4], init.clone(), chunk_size=4)
    (a_out.sum() + a_state.square().sum()).backward()
    grads_chunk = [t.grad.clone() for t in leaves]
    for t in leaves:
        t.grad = None
    s = init.clone()
    outs = []
    for t in range(11):
        s = s * leaves[3][:, :, t].unsqueeze(-2)
        err = leaves[2][:, :, t] - torch.einsum("bhvd,bhd->bhv", s, leaves[1][:, :, t])
        s = s + leaves[4][:, :, t][:, :, None, None] * err.unsqueeze(-1) * leaves[1][:, :, t].unsqueeze(-2)
        outs.append(torch.einsum("bhvd,bhd->bhv", s, leaves[0][:, :, t]))
    (torch.stack(outs, 2).sum() + s.square().sum()).backward()
    for g_chunk, t in zip(grads_chunk, leaves):
        assert torch.allclose(g_chunk, t.grad, atol=1e-8, rtol=1e-8)
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
