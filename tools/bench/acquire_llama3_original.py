"""Acquire a pinned original BF16 Llama reference; no remote code or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import time
import urllib.request

META = 'meta-llama/Llama-3.2-1B-Instruct'
META_REVISION = '9213176726f574b556790deb65791e0c5aa438b6'
MIRROR = 'alpindale/Llama-3.2-1B-Instruct'
REVISION = 'f92201d8185818a9d079b3b52efdab4b68bdd17f'
WEIGHT_BYTES = 2471645608
WEIGHT_SHA = '1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f'
BLOBS = {
    'config.json': '3e3aaf51a035cb5092d9f6827a0dc074657ba88c',
    'tokenizer.json': '5cc5f00a5b203e90a27a3bd60d1ec393b07971e8',
    'tokenizer_config.json': '4ff488a165e900e5129cda7c20ab32d568d2a475',
    'special_tokens_map.json': '02ee80b6196926a5ad790a004d9efd6ab1ba6542',
}
POINTER = '5daf23edd80cedb879a4a18e6049a19b2417a3f7'


def git_blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def inventory(document, revision):
    if document.get('sha') != revision:
        raise ValueError('Reference revision differs')
    rows = document['siblings']
    files = {row['rfilename']: row for row in rows}
    if len(files) != len(rows):
        raise ValueError('Duplicate inventory file')
    for name, blob in BLOBS.items():
        if files[name]['blobId'] != blob or not 0 < files[name]['size'] <= 16 * 1024**2:
            raise ValueError('Original metadata identity differs: ' + name)
    weight = files['model.safetensors']
    if (weight['size'] != WEIGHT_BYTES or weight['blobId'] != POINTER
            or weight['lfs']['sha256'] != WEIGHT_SHA or weight['lfs']['size'] != WEIGHT_BYTES):
        raise ValueError('Original checkpoint identity differs')
    return files


def transfer(url, path, maximum, expected_bytes=None, expected_sha=None, disk_reserve=0, context=None):
    """One exclusive bounded sink; partial failures are retained for inspection."""
    deadline = time.monotonic() + 1800
    digest = hashlib.sha256()
    count = 0
    request = urllib.request.Request(url, headers={'User-Agent': 'toklet-reference-acquisition/1', 'Cache-Control': 'no-cache'})
    def space():
        if disk_reserve and (expected_bytes is None or shutil.disk_usage(path.parent).free < expected_bytes - count + disk_reserve):
            raise ValueError('Remaining payload plus disk reserve no longer fits')
    space()
    checked = 0
    with path.open('xb') as output:
        with urllib.request.urlopen(request, context=context or ssl.create_default_context(), timeout=30) as response:
            if response.status != 200 or not response.geturl().startswith('https://'):
                raise ValueError('Expected complete TLS response')
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Reference transfer deadline')
                chunk = response.read(1024**2)
                if not chunk:
                    break
                count += len(chunk)
                if count > maximum:
                    raise ValueError('Reference transfer exceeds bound')
                output.write(chunk)
                digest.update(chunk)
                if count - checked >= 8 * 1024**2:
                    output.flush()
                    space()
                    checked = count
    actual = digest.hexdigest()
    if ((expected_bytes is not None and count != expected_bytes)
            or (expected_sha is not None and actual != expected_sha)):
        raise ValueError('Reference payload size or SHA-256 differs')
    return {'bytes': count, 'sha256': actual, 'url': url}


def acquire(directory, ca_file=None):
    context = ssl.create_default_context(cafile=ca_file)
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError('Fresh reference directory required')
    if shutil.disk_usage(directory.parent).free < WEIGHT_BYTES + 1024**3:
        raise ValueError('Reference file plus 1 GiB disk reserve required')
    directory.mkdir()
    result = {'format': 'toklet-llama3-original-acquisition-v1', 'passed': False,
              'original': {'repository': META, 'revision': META_REVISION},
              'mirror': {'repository': MIRROR, 'revision': REVISION}, 'files': {},
              'tls_ca_sha256': hashlib.sha256(Path(ca_file).read_bytes()).hexdigest() if ca_file else None}
    try:
        for label, repository, revision in [('meta', META, META_REVISION), ('mirror', MIRROR, REVISION)]:
            url = f'https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true'
            path = directory / (label + '-inventory.json')
            result[label + '_inventory'] = transfer(url, path, 2 * 1024**2, context=context)
            document = json.loads(path.read_bytes())
            files = inventory(document, revision)
            if label == 'mirror':
                mirror_files = files
        for name, blob in BLOBS.items():
            path = directory / name
            row = transfer(f'https://huggingface.co/{MIRROR}/resolve/{REVISION}/{name}?download=true',
                           path, 16 * 1024**2, expected_bytes=mirror_files[name]['size'], context=context)
            if git_blob(path.read_bytes()) != blob:
                raise ValueError('Original Git blob differs: ' + name)
            row['git_blob'] = blob
            result['files'][name] = row
        partial = directory / 'model.safetensors.partial'
        row = transfer(f'https://huggingface.co/{MIRROR}/resolve/{REVISION}/model.safetensors?download=true',
                       partial, WEIGHT_BYTES, WEIGHT_BYTES, WEIGHT_SHA, disk_reserve=1024**3, context=context)
        # Exclusive directory and no overwriting publication. No second payload copy.
        destination = directory / 'model.safetensors'
        destination.hardlink_to(partial)
        partial.unlink()
        result['files']['model.safetensors'] = row
        result['passed'] = True
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        (directory / 'acquisition.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--ca-file', type=Path, help='Trusted CA bundle; certificate/hostname verification stay enabled')
    args = parser.parse_args()
    print(json.dumps(acquire(args.directory, args.ca_file), indent=2))
