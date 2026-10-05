"""Full-parameter MiniK3 functional smoke using every real pretraining source.

This does not establish long-context quality or full-length four-GPU capacity.

Every optimizer step accumulates one micro-batch from *every* source, so all
sources go through forward, backward and the update. Training steps use the
production path ``compute_logits=False`` (chunked LM + MTP loss).

``SMOKE.json`` binding: the report records the manifest/AUDIT/shard hashes and
``model_code_sha256`` (``train/model_fingerprint.py``); ``readiness.py`` refuses
to train when the current model code differs. Progress is written to
``<report>.running.json``; the report itself is replaced atomically only when the
smoke passes, so a failed rerun never destroys a previously passed SMOKE.json
(the failure is kept in ``<report>.failed.json``).
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from train.config import MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM
from train.engine.balancer import NoAuxBalancer
from train.engine.muon import build_optimizer
from train.engine.init_patch import assert_initialised
from train.data.smoke_audit import verified_smoke_manifest
from train.model_fingerprint import model_code_fingerprint


def sidecar(report: Path, tag: str) -> Path:
    return report.with_name(f"{report.stem}.{tag}{report.suffix or '.json'}")


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--manifest',required=True)
    p.add_argument('--report',required=True,
                   help='Report path; readiness reads <manifest dir>/SMOKE.json. Replaced only on success.')
    p.add_argument('--sequence-length',type=int,default=64)
    p.add_argument('--steps',type=int,default=2,help='Optimizer steps; each uses one batch from every source.')
    a=p.parse_args()
    report=Path(a.report);report.parent.mkdir(parents=True,exist_ok=True)
    running=sidecar(report,'running');failed=sidecar(report,'failed')
    code_sha=model_code_fingerprint()
    status={'status':'running','sequence_length':a.sequence_length,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'model_code_sha256':code_sha,
            'scope':'full model, real data from all sources, every optimizer step accumulates one batch per source; '
                    'single GPU functional check'}
    write_json_atomic(running,status)
    try:
        cfg=MiniK3Config();cfg.validate()
        data,evidence=verified_smoke_manifest(a.manifest,cfg.stable_mix,cfg.vocab_size)
        status.update(evidence)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for production-model smoke')
        torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
        device=torch.device('cuda');model=MiniK3ForCausalLM(cfg).to(device);assert_initialised(model)
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=build_optimizer(
            iter(trainable), lr=1e-5, weight_decay=cfg.weight_decay,
            muon_update_scale=cfg.muon_update_scale,
        )
        # V100 FP16 needs explicit unscale before clipping.  Clipping scaled
        # gradients was the cause of the earlier false non-finite smoke.
        scaler=torch.amp.GradScaler('cuda', init_scale=1024.0, growth_interval=2000)
        balancer=NoAuxBalancer(model)
        batches={}
        for source,info in data['sources'].items():
            path=info['shards'][0];tokens=np.fromfile(path,dtype='<u4',count=a.sequence_length)
            if len(tokens)!=a.sequence_length:raise ValueError('Shard too short for smoke: '+source)
            batches[source]=torch.from_numpy(tokens.astype(np.int64)).to(device).unsqueeze(0)
        losses={};mtp_losses={}
        model.eval()
        for source,x in batches.items():
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                out=model(x,labels=x,compute_logits=False)
            value=float(out['lm_loss'])
            if not math.isfinite(value):raise ValueError('Non-finite source loss: '+source)
            losses[source]=value
            if out.get('mtp_loss') is not None:mtp_losses[source]=float(out['mtp_loss'])
        if max(abs(x-math.log(cfg.vocab_size)) for x in losses.values())>2.0:
            raise ValueError('Initial loss far from vocabulary baseline; initialization/labels require investigation')
        torch.cuda.reset_peak_memory_stats(device)
        model.train();steps=[]
        n=len(batches)
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            step_loss=step_lm=0.0;step_mtp=[]
            for source,x in batches.items():
                with torch.autocast('cuda',dtype=torch.float16):
                    out=model(x,labels=x,compute_logits=False)
                loss=out['loss']
                if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite training loss on {source}')
                scaler.scale(loss/n).backward()
                step_loss+=float(loss)/n;step_lm+=float(out['lm_loss'])/n
                if out.get('mtp_loss') is not None:step_mtp.append(float(out['mtp_loss']))
                del out,loss
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.0)
            if not torch.isfinite(norm):
                nonfinite=[name for name,p in model.named_parameters()
                           if p.grad is not None and not torch.isfinite(p.grad).all()]
                raise FloatingPointError('Non-finite gradients: ' + str(nonfinite[:12]))
            unused=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None and '.experts.' not in name and not name.startswith('vision.')]
            if unused:raise ValueError('Unused non-expert trainable parameters: '+str(unused[:12]))
            scaler.step(optimizer);scaler.update();telemetry=balancer.step()
            steps.append({'loss':step_loss,'lm_loss':step_lm,
                          'mtp_loss':(sum(step_mtp)/len(step_mtp)) if step_mtp else None,
                          'sources':list(batches),'grad_norm':float(norm),'router':telemetry})
        assert_initialised(model)
        if model_code_fingerprint()!=code_sha:
            raise RuntimeError('Model code changed while the smoke was running; re-run smoke_from_manifest.py')
        status.update(status='passed',source_losses=losses,source_mtp_losses=mtp_losses,steps=steps,
                      parameters=model.count_parameters(),gpu=torch.cuda.get_device_name(),
                      peak_memory_bytes=torch.cuda.max_memory_allocated(),
                      peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
                      finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        write_json_atomic(report,status)
        running.unlink(missing_ok=True);failed.unlink(missing_ok=True)
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if torch.cuda.is_available():
            status.update(peak_memory_bytes=torch.cuda.max_memory_allocated(),
                          peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json_atomic(failed,status)
        running.unlink(missing_ok=True)
        print(f'[smoke] FAILED; {report} left unchanged; details in {failed}',file=sys.stderr)
        raise

if __name__=='__main__':main()
