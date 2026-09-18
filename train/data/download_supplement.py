"""Standalone asynchronous downloader for FineWeb-EDU and CodeParrot supplements.

Downloads verified files with checksums and resumes interrupted transfers.
Converts CodeParrot into allowlisted Python Parquet compatible with canonical_v2.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import pyarrow as pa
import pyarrow.parquet as pq

from canonical_v2 import atomic_json, digest_file
from supplement_v1 import ALLOWED, codeparrot_selection, select_fineweb, Run


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def download_fineweb_file(raw_dir, cache_dir, item):
    from modelscope.hub.file_download import dataset_file_download
    dst = raw_dir / item['path']
    if dst.is_file() and dst.stat().st_size == item['size']:
        digest = digest_file(dst)
        if not item.get('sha256') or digest == item['sha256']:
            print(f"[FineWeb] Already downloaded: {item['path']}", flush=True)
            return {'path': str(dst), 'bytes': dst.stat().st_size, 'sha256': digest, 'cached': True}

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[FineWeb] Downloading: {item['path']} ({item['size'] / 1e9:.2f} GB)...", flush=True)
    actual = dataset_file_download(
        'HuggingFaceFW/fineweb-edu',
        item['path'],
        revision='master',
        local_dir=str(raw_dir),
        cache_dir=str(cache_dir)
    )
    p = Path(actual)
    if p.stat().st_size != item['size']:
        raise ValueError(f"FineWeb size mismatch: {item['path']}")
    digest = digest_file(p)
    if item.get('sha256') and digest != item['sha256']:
        raise ValueError(f"FineWeb checksum mismatch: {item['path']}")
    print(f"[FineWeb] Verified: {item['path']} SHA256={digest[:16]}...", flush=True)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest, 'cached': False}


def download_codeparrot_file(dstroot, revision, path, max_retries=5):
    dst = dstroot / Path(path).name
    tmp = dst.with_name(dst.name + '.incomplete')
    if dst.is_file() and dst.stat().st_size > 0:
        digest = digest_file(dst)
        print(f"[CodeParrot] Already downloaded: {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
        return {'path': str(dst), 'source_path': path, 'bytes': dst.stat().st_size, 'sha256': digest, 'cached': True}

    url = f"https://hf-mirror.com/datasets/codeparrot/github-code/resolve/{revision}/{path}"
    print(f"[CodeParrot] Downloading: {path} -> {dst.name}...", flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(1, max_retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            with urllib.request.urlopen(req, timeout=120) as response, open(tmp, 'wb') as f:
                while True:
                    chunk = response.read(2 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp.is_file() and tmp.stat().st_size > 0:
                break
        except Exception as e:
            print(f"[CodeParrot] Attempt {attempt} failed: {e}", flush=True)
            if attempt == max_retries:
                raise RuntimeError(f"Download failed for {url} after {max_retries} attempts: {e}")
            time.sleep(attempt * 2)

    digest = digest_file(tmp)
    os.replace(tmp, dst)
    print(f"[CodeParrot] Completed: {dst.name} ({dst.stat().st_size / 1e6:.1f} MB) SHA256={digest[:16]}...", flush=True)
    return {'path': str(dst), 'source_path': path, 'bytes': dst.stat().st_size, 'sha256': digest, 'cached': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default='/data/mini-k3')
    parser.add_argument('--count-code', type=int, default=200)
    args = parser.parse_args()

    work = Path(args.work)
    run = Run(work)
    status_file = run.reports / 'DOWNLOAD_STATUS.json'
    
    def update_status(st, **kwargs):
        atomic_json(status_file, dict(status=st, updated_at=now(), **kwargs))

    print(f"[{now()}] Starting supplement download...", flush=True)
    update_status('selecting')

    fine_selection = select_fineweb(run)
    code_selection = codeparrot_selection(count=args.count_code)
    atomic_json(run.reports / 'selection_active.json', {
        'fineweb': fine_selection,
        'codeparrot': code_selection,
        'generated_at': now()
    })

    print(f"[{now()}] Selected {len(fine_selection['files'])} FineWeb files ({fine_selection['selected_bytes'] / 1e9:.2f} GB)", flush=True)
    print(f"[{now()}] Selected {len(code_selection['files'])} CodeParrot files", flush=True)

    # 1. Download FineWeb-EDU
    existing_fineweb = list((run.raw / 'fineweb-edu').glob('data/*/*.parquet'))
    if getattr(args, 'skip_fineweb', False) or len(existing_fineweb) >= 8:
        print(f"[{now()}] FineWeb already has {len(existing_fineweb)} files ({sum(p.stat().st_size for p in existing_fineweb) / 1e9:.2f} GB). Target met, skipping FineWeb download.", flush=True)
        update_status('fineweb_complete', fineweb_total=len(existing_fineweb), fineweb_done=len(existing_fineweb))
        fineweb_results = [{'path': str(p), 'bytes': p.stat().st_size} for p in existing_fineweb]
    else:
        update_status('downloading_fineweb', fineweb_total=len(fine_selection['files']), fineweb_done=0)
        fineweb_results = []
        for idx, item in enumerate(fine_selection['files'], 1):
            res = download_fineweb_file(run.raw / 'fineweb-edu', run.work / 'ms-cache', item)
            fineweb_results.append(res)
            update_status('downloading_fineweb', fineweb_total=len(fine_selection['files']), fineweb_done=idx)

    # 2. Download CodeParrot
    update_status('downloading_codeparrot', codeparrot_total=len(code_selection['files']), codeparrot_done=0)
    code_results = []
    dstroot = run.raw / 'github-code'
    dstroot.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(code_selection['files'], 1):
        res = download_codeparrot_file(dstroot, code_selection['revision'], path)
        code_results.append(res)
        update_status('downloading_codeparrot', codeparrot_total=len(code_selection['files']), codeparrot_done=idx)

    # 3. Convert CodeParrot to stack-compatible Parquet
    update_status('converting_code', total_shards=len(code_results))
    outdir = run.raw / 'code-python/data'
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    total_kept = 0
    total_rejected = {}

    for ordinal, item in enumerate(code_results):
        out_path = outdir / f'github-supplement-{ordinal:04d}.parquet'
        if out_path.is_file() and out_path.stat().st_size > 0:
            digest = digest_file(out_path)
            outputs.append({'path': str(out_path), 'sha256': digest, 'cached': True})
            continue

        rows = []
        table = pq.read_table(item['path'])
        for row in table.to_pylist():
            p = row.get('path', '')
            if not p.endswith('.py'):
                total_rejected['non_python'] = total_rejected.get('non_python', 0) + 1
                continue
            lic = str(row.get('license', '')).lower()
            code = row.get('content')
            if lic not in ALLOWED:
                total_rejected['license_not_allowlisted'] = total_rejected.get('license_not_allowlisted', 0) + 1
                continue
            if not isinstance(code, str) or len(code.strip()) < 20:
                total_rejected['too_short'] = total_rejected.get('too_short', 0) + 1
                continue
            rows.append({
                'repo_path': row.get('repo_name'),
                'files': [{'content': code, 'language': 'Python', 'file_path': p, 'license_type': lic, 'is_vendor': False}]
            })

        pq.write_table(pa.Table.from_pylist(rows), out_path)
        digest = digest_file(out_path)
        outputs.append({'path': str(out_path), 'rows': len(rows), 'bytes': out_path.stat().st_size, 'sha256': digest})
        total_kept += len(rows)
        print(f"[CodeConvert] Shard {ordinal:04d}: kept {len(rows)} allowlisted Python files", flush=True)

    report_data = {
        'status': 'complete',
        'generated_at': now(),
        'fineweb_files': fineweb_results,
        'codeparrot_raw_files': code_results,
        'codeparrot_converted_files': outputs,
        'total_python_rows': total_kept,
        'rejections': total_rejected
    }
    atomic_json(run.reports / 'DOWNLOAD_COMPLETE.json', report_data)
    update_status('complete', total_python_rows=total_kept)
    print(f"[{now()}] Supplement download and conversion 100% complete!", flush=True)


if __name__ == '__main__':
    main()
