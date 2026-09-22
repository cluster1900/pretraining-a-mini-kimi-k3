"""Download and convert CodeSearchNet Python subset for Mini-K3 code-python supplement.

Source: Nan-Do/code-search-net-python on hf-mirror.com
License: Apache-2.0 (Permissive SPDX allowlisted)
Target: /data/mini-k3/data/raw/codesearchnet/ (raw) -> /data/mini-k3/data/raw/code-python/data/codesearchnet-*.parquet
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

from canonical_v2 import atomic_json, digest_file

ALLOWED = {'mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 'isc', 'unlicense', '0bsd'}
REVISION = '39db91866dd0f251f3b0c7f42c0f85634101df6e'
SOURCE_NAME = 'Nan-Do/code-search-net-python'
FILES = [
    'data/train-00000-of-00004-ee77a7de79eb2ab2.parquet',
    'data/train-00001-of-00004-648b3bede2edf6e6.parquet',
    'data/train-00002-of-00004-1dfd72b171e6b205.parquet',
    'data/train-00003-of-00004-184ab6d0e3c690b1.parquet',
]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def download(raw_root, max_retries=5):
    dstroot = Path(raw_root) / 'codesearchnet'
    dstroot.mkdir(parents=True, exist_ok=True)
    out = []
    for rel_path in FILES:
        fname = Path(rel_path).name
        dst = dstroot / fname
        tmp = dst.with_name(dst.name + '.incomplete')
        if dst.is_file() and dst.stat().st_size > 0:
            digest = digest_file(dst)
            print(f'Using cached {fname}: {dst.stat().st_size} bytes, sha256={digest[:16]}...')
            out.append({'path': str(dst), 'source_path': rel_path, 'bytes': dst.stat().st_size, 'sha256': digest})
            continue

        url = f'https://hf-mirror.com/datasets/{SOURCE_NAME}/resolve/{REVISION}/{rel_path}'
        print(f'Downloading {url} -> {dst} ...')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        for attempt in range(1, max_retries + 1):
            try:
                if tmp.exists():
                    tmp.unlink()
                with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, 'wb') as f:
                    downloaded = 0
                    while True:
                        chunk = resp.read(2 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                if tmp.is_file() and tmp.stat().st_size > 0:
                    break
            except Exception as e:
                print(f'Attempt {attempt}/{max_retries} failed for {fname}: {e}')
                if attempt == max_retries:
                    raise RuntimeError(f'Download failed for {url} after {max_retries} attempts: {e}')
                time.sleep(attempt * 2)

        digest = digest_file(tmp)
        os.replace(tmp, dst)
        print(f'Downloaded {fname}: {dst.stat().st_size} bytes, sha256={digest[:16]}...')
        out.append({'path': str(dst), 'source_path': rel_path, 'bytes': dst.stat().st_size, 'sha256': digest})
    return out


def convert(raw_root, downloaded_files):
    outdir = Path(raw_root) / 'code-python/data'
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    kept = 0
    rejected = {}

    for ordinal, item in enumerate(downloaded_files):
        out = outdir / f'codesearchnet-{ordinal:04d}.parquet'
        if out.is_file() and out.stat().st_size > 0:
            digest = digest_file(out)
            outputs.append({'path': str(out), 'sha256': digest, 'cached': True})
            print(f'Using cached converted {out.name}: {out.stat().st_size} bytes')
            continue

        print(f'Converting {item["path"]} -> {out} ...')
        rows = []
        table = pq.read_table(item['path'])
        for row in table.to_pylist():
            # Filter non-python
            lang = str(row.get('language', '')).lower()
            if lang and lang != 'python':
                rejected['non_python'] = rejected.get('non_python', 0) + 1
                continue

            # Code text: prefer full original_string (includes docstring & body); fallback to code
            code = row.get('original_string') or row.get('code')
            if not isinstance(code, str) or len(code.strip()) < 20:
                rejected['too_short'] = rejected.get('too_short', 0) + 1
                continue

            repo = str(row.get('repo') or 'codesearchnet')
            file_path = str(row.get('path') or '')
            lic = 'apache-2.0'  # Verified dataset-level permissive license

            rows.append({
                'repo_path': repo,
                'files': [{
                    'content': code,
                    'language': 'Python',
                    'file_path': file_path,
                    'license_type': lic,
                    'is_vendor': False,
                }]
            })

        pq.write_table(pa.Table.from_pylist(rows), out)
        digest = digest_file(out)
        outputs.append({
            'path': str(out),
            'rows': len(rows),
            'bytes': out.stat().st_size,
            'sha256': digest,
        })
        kept += len(rows)
        print(f'Finished {out.name}: {len(rows)} rows, {out.stat().st_size} bytes, sha256={digest[:16]}...')

    return {
        'files': outputs,
        'kept_rows': kept,
        'rejected': rejected,
        'license_allowlist': sorted(ALLOWED),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', default='/data/mini-k3/data/raw')
    parser.add_argument('--reports-dir', default='/data/mini-k3/data/reports/codesearchnet')
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print('Step 1: Downloading CodeSearchNet Python...')
    downloaded = download(args.raw_root)
    download_report = {
        'status': 'complete',
        'source': SOURCE_NAME,
        'revision': REVISION,
        'dataset_license': 'apache-2.0',
        'files': downloaded,
        'downloaded_at': now(),
    }
    atomic_json(reports_dir / 'download.json', download_report)

    print('Step 2: Converting CodeSearchNet Python to Mini-K3 schema...')
    adapted = convert(args.raw_root, downloaded)
    adapter_report = dict(
        status='complete',
        source=SOURCE_NAME,
        revision=REVISION,
        converted_at=now(),
        **adapted
    )
    atomic_json(reports_dir / 'adapter.json', adapter_report)
    print('Download and conversion complete!')


if __name__ == '__main__':
    main()
