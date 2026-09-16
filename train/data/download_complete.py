"""Full repository inventories and resumable, verified server-side downloads.

Never equates a byte quota or one API page with repository completion.
No cleaning, tokenization or training is performed by this program.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote, urlsplit, urlunsplit

MIRROR = "https://hf-mirror.com"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_path(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Unsafe repository path')
    result = Path(root).joinpath(*rel.parts)
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes source directory')
    return result


def mirror_url(url):
    u = urlsplit(url)
    if u.hostname not in ('huggingface.co', 'hf-mirror.com'):
        raise ValueError('Unexpected pagination host')
    return urlunsplit(('https', 'hf-mirror.com', u.path, u.query, ''))


def api_get(url):
    # Header URLs are used internally only; never print signed download URLs.
    with tempfile.TemporaryDirectory(prefix='api-') as td:
        headers, body = Path(td)/'headers', Path(td)/'body'
        cp = subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '2',
            '--connect-timeout', '15', '--max-time', '90', '-A', 'Mozilla/5.0',
            '-D', str(headers), '-o', str(body), url], capture_output=True)
        if cp.returncode:
            raise RuntimeError('API request failed: ' + cp.stderr.decode(errors='replace')[-350:])
        data = json.loads(body.read_bytes())
        links = re.findall(r'<([^>]+)>;\s*rel="?next"?', headers.read_text(), re.I)
        return data, mirror_url(links[-1]) if links else None


def inventory(source, root):
    target = root/source['directory']/'INVENTORY.json'
    state = root.parent/'reports'/'downloads'/f"{source['name']}.json"
    atomic(state, dict(source, status='listing', updated_at=now()))
    repo = source['repo']
    files = []
    paths = set()
    try:
        if source['backend'] == 'hf-mirror':
            meta, _ = api_get(f'{MIRROR}/api/datasets/{repo}')
            revision = meta['sha']
            license_name = (meta.get('cardData') or {}).get('license')
            gated = meta.get('gated', False)
            url = f'{MIRROR}/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000'
            urls = set()
            while url:
                if url in urls:
                    raise ValueError('Repeated pagination URL')
                urls.add(url)
                page, url = api_get(url)
                for item in page:
                    if item.get('type') != 'file':
                        continue
                    path = item['path']
                    if path in paths:
                        raise ValueError('Duplicate file in paginated listing')
                    paths.add(path)
                    files.append({'path': path, 'size': int(item['size']),
                        'sha256': (item.get('lfs') or {}).get('oid'),
                        'git_oid': item.get('oid') if not item.get('lfs') else None})
        else:
            from modelscope.hub.api import HubApi
            api = HubApi()
            meta = api.get_repo(repo, repo_type='dataset')
            license_name, gated = meta.license, meta.gated
            revision = 'master'
            # Legacy API defaults to only 100 entries. Explicitly exhaust pages.
            page_no = 1
            fingerprints = set()
            while True:
                page = api.get_dataset_files(repo, revision=revision, recursive=True,
                                             page_number=page_no, page_size=100)
                if not page:
                    break
                fingerprint = tuple(x['Path'] for x in page)
                if fingerprint in fingerprints:
                    raise ValueError('Server repeated a page; inventory is incomplete')
                fingerprints.add(fingerprint)
                for item in page:
                    if item.get('Type') != 'blob':
                        continue
                    path = item['Path']
                    if path in paths:
                        raise ValueError('Duplicate file in paginated listing')
                    paths.add(path)
                    files.append({'path': path, 'size': int(item.get('Size') or 0),
                        'sha256': item.get('Sha256') or None,
                        'file_revision': item.get('Revision') or None})
                page_no += 1
                if len(page) < 100:
                    break
        if not files:
            raise ValueError('Repository contains no files')
        for f in files:
            safe_path(root/source['directory'], f['path'])
        files.sort(key=lambda x: x['path'])
        inv = dict(source, revision=revision, license=license_name, gated=gated,
                   listed_at=now(), files=files, total_bytes=sum(f['size'] for f in files))
        inv['content_manifest_sha256'] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        atomic(target, inv)
        existing = sum(f['size'] for f in files if safe_path(root/source['directory'], f['path']).is_file()
                       and safe_path(root/source['directory'], f['path']).stat().st_size == f['size'])
        atomic(state, dict(source, status='listed', inventory=str(target), files=len(files),
                          total_bytes=inv['total_bytes'], present_size_matched_bytes=existing,
                          remaining_bytes=inv['total_bytes']-existing, license=license_name,
                          gated=gated, updated_at=now()))
        print(json.dumps({'source':source['name'], 'status':'listed', 'files':len(files),
                          'bytes':inv['total_bytes']}, ensure_ascii=False), flush=True)
    except Exception as exc:
        atomic(state, dict(source, status='listing_failed', error=str(exc)[:500], updated_at=now()))
        print(json.dumps({'source':source['name'], 'status':'listing_failed', 'error':str(exc)[:200]}), flush=True)


def checksum(path, item):
    sha = hashlib.sha256()
    git = hashlib.sha1()
    git.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            sha.update(block)
            git.update(block)
    digest = sha.hexdigest()
    if item.get('sha256') and digest != item['sha256']:
        raise ValueError('Upstream SHA256 mismatch: ' + item['path'])
    if item.get('git_oid') and git.hexdigest() != item['git_oid']:
        raise ValueError('Upstream Git blob hash mismatch: ' + item['path'])
    return digest


def fetch(inv, item, directory):
    dst = safe_path(directory, item['path'])
    if dst.is_file() and dst.stat().st_size == item['size']:
        try:
            return dst, checksum(dst, item)
        except ValueError:
            # Preserve a corrupt existing file for diagnosis; do not reuse it.
            dst.rename(dst.with_name(dst.name + '.corrupt-' + str(time.time_ns())))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if inv['backend'] == 'modelscope':
        from modelscope.hub.file_download import dataset_file_download
        actual = dataset_file_download(inv['repo'], item['path'], revision=inv['revision'],
            local_dir=str(directory), cache_dir=str(directory.parent.parent.parent/'ms-cache'))
        if Path(actual).resolve() != dst.resolve():
            raise ValueError('Unexpected local download path')
    else:
        tmp = dst.with_name(dst.name+'.incomplete')
        url = f"{MIRROR}/datasets/{inv['repo']}/resolve/{inv['revision']}/{quote(item['path'], safe='/')}"
        cp = subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '5', '--retry-delay', '3',
            '--connect-timeout', '20', '--max-time', '43200', '--speed-limit', '1024',
            '--speed-time', '180', '-A', 'Mozilla/5.0', '-C', '-', '-o', str(tmp), url])
        if cp.returncode:
            raise RuntimeError('curl failed with exit ' + str(cp.returncode))
        if tmp.stat().st_size != item['size']:
            raise ValueError('Downloaded size mismatch: ' + item['path'])
        checksum(tmp, item)
        os.replace(tmp, dst)
    if dst.stat().st_size != item['size']:
        raise ValueError('Downloaded size mismatch: ' + item['path'])
    return dst, checksum(dst, item)


def download(source, root, reserve):
    directory = root/source['directory']
    state_path = root.parent/'reports'/'downloads'/f"{source['name']}.json"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.full-download.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        inv = json.loads((directory/'INVENTORY.json').read_text())
        state = dict(source, status='downloading', total_bytes=inv['total_bytes'],
                     files=len(inv['files']), verified_files=0, verified_bytes=0, updated_at=now())
        receipt = {'inventory_sha256':inv['content_manifest_sha256'], 'repo':inv['repo'],
                   'revision':inv['revision'], 'license':inv['license'], 'files':[]}
        atomic(state_path, state)
        try:
            for item in inv['files']:
                # No global installs or SSD caches; keep disk space for logs/checkpoints.
                if shutil.disk_usage(root).free < item['size'] + reserve:
                    raise OSError('Insufficient /data space; reserve preserved')
                state.update(current_file=item['path'], current_file_bytes=item['size'], updated_at=now())
                atomic(state_path, state)
                for attempt in range(3):
                    try:
                        dst, digest = fetch(inv, item, directory)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(3*(attempt+1))
                receipt['files'].append(dict(item, local=str(dst), actual_sha256=digest))
                state['verified_files'] += 1
                state['verified_bytes'] += item['size']
                atomic(directory/'VERIFIED_FILES.json', receipt)
                atomic(state_path, dict(state, updated_at=now()))
                print(json.dumps({'source':source['name'], 'file':item['path'],
                                  'verified_bytes':state['verified_bytes']},ensure_ascii=False), flush=True)
            # This marker is different from the old quota-based DOWNLOAD.json.
            atomic(directory/'FULL_DOWNLOAD_COMPLETE.json', dict(receipt, completed_at=now(),
                    total_bytes=state['verified_bytes'], files_count=state['verified_files']))
            atomic(state_path, dict(state, status='complete', updated_at=now()))
        except Exception as exc:
            atomic(state_path, dict(state, status='failed', error=str(exc)[:500], updated_at=now()))
            print(json.dumps({'source':source['name'],'status':'failed','error':str(exc)[:250]}),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--mode', choices=['inventory','download','all'], required=True)
    p.add_argument('--sources', nargs='*')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    catalog = json.loads(a.catalog.read_text())
    if catalog.get('download_scope') == 'usage_only' and a.mode in ('download', 'all'):
        raise SystemExit('Full repository downloads are disabled by the usage-only policy. Build a file selection from the training manifest/token deficit first.')
    root = Path(catalog['root'])
    if not root.resolve().is_relative_to('/data/mini-k3'):
        raise ValueError('Bulk data must remain under /data/mini-k3')
    root.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = '/data/mini-k3/tmp'
    Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    tempfile.tempdir = os.environ['TMPDIR']
    os.environ['MODELSCOPE_CACHE'] = '/data/mini-k3/ms-cache'
    sources = [s for s in catalog['sources'] if not a.sources or s['name'] in a.sources]
    if a.mode in ('inventory','all'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(lambda s: inventory(s,root), sources))
    if a.mode in ('download','all'):
        # Reserve aggregate space before launching competing jobs.
        budget = shutil.disk_usage(root).free - catalog['reserve_bytes']
        accepted = []
        decisions = []
        for source in sources:
            path = root/source['directory']/'INVENTORY.json'
            status_file = root.parent/'reports'/'downloads'/f"{source['name']}.json"
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            if not path.exists() or status.get('status') == 'listing_failed':
                decisions.append(dict(source,status='no_inventory'))
                continue
            inv=json.loads(path.read_text())
            remaining=sum(f['size'] for f in inv['files'] if not safe_path(root/source['directory'],f['path']).is_file()
                or safe_path(root/source['directory'],f['path']).stat().st_size != f['size'])
            if remaining>budget:
                decision=dict(source,status='blocked_capacity',remaining_bytes=remaining,
                              available_unreserved_bytes=budget,updated_at=now())
                atomic(status_file, decision)
                decisions.append(decision)
                continue
            budget-=remaining
            accepted.append(source)
            decisions.append(dict(source,status='queued',remaining_bytes=remaining))
        atomic(root.parent/'reports'/'DOWNLOAD_QUEUE.json',dict(updated_at=now(),jobs=decisions))
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(lambda s: download(s,root,catalog['reserve_bytes']),accepted))


if __name__=='__main__':
    main()
