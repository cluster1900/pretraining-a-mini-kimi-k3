"""Full-parameter MiniK3 functional smoke using every real pretraining source.

This does not establish long-context quality or full-length four-GPU capacity.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from train.config import MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM
from train.engine.balancer import NoAuxBalancer
from train.engine.init_patch import assert_initialised
from train.data.smoke_audit import verified_smoke_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--report',required=True)
    p.add_argument('--sequence-length',type=int,default=64);a=p.parse_args()
    report=Path(a.report);report.parent.mkdir(parents=True,exist_ok=True)
    status={'status':'running','sequence_length':a.sequence_length,'scope':'full model, real data from all sources, two optimizer steps; single GPU functional check'}
    report.write_text(json.dumps(status,indent=2))
    try:
        cfg=MiniK3Config();cfg.validate()
        data,evidence=verified_smoke_manifest(a.manifest,cfg.stable_mix,cfg.vocab_size)
        status.update(evidence)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for production-model smoke')
        torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
        device=torch.device('cuda');model=MiniK3ForCausalLM(cfg).to(device);assert_initialised(model)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.95),foreach=False)
        scaler=torch.amp.GradScaler('cuda');balancer=NoAuxBalancer(model)
        losses={}
        for source,info in data['sources'].items():
            path=info['shards'][0];tokens=np.fromfile(path,dtype='<u4',count=a.sequence_length)
            if len(tokens)!=a.sequence_length:raise ValueError('Shard too short for smoke')
            x=torch.tensor(tokens.astype(np.int64),device=device).unsqueeze(0)
            model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):loss=model(x,labels=x)['loss']
            value=float(loss)
            if not math.isfinite(value):raise ValueError('Non-finite source loss: '+source)
            losses[source]=value
        if max(abs(x-math.log(cfg.vocab_size)) for x in losses.values())>2.0:
            raise ValueError('Initial loss far from vocabulary baseline; initialization/labels require investigation')
        model.train();steps=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=model(x,labels=x)['loss']
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            unused=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None and '.experts.' not in n]
            if unused:raise ValueError('Unused non-expert trainable parameters: '+str(unused[:12]))
            scaler.step(optimizer);scaler.update();telemetry=balancer.step()
            steps.append({'loss':float(loss),'grad_norm':float(norm),'router':telemetry})
        assert_initialised(model)
        status.update(status='passed',source_losses=losses,steps=steps,parameters=model.count_parameters(),
                      gpu=torch.cuda.get_device_name(),peak_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        status.update(status='failed',error=str(exc));raise
    finally:report.write_text(json.dumps(status,indent=2))

if __name__=='__main__':main()
