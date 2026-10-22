"""Static inventory of pinned upstream source archives; never imports their code.

Entry points are observable exports at the loader's model-module boundary, not
proof that an upstream configuration works or that Toklet implements its graph.
"""
import argparse
import ast
import hashlib
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

MAX_ARCHIVE = 64 * 1024**2
MAX_EXPANDED = 256 * 1024**2
MAX_SOURCE = 2 * 1024**2
MAX_MEMBERS = 20000


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def symbols(raw, name):
    """Read top-level bindings without evaluating annotations or imports."""
    tree = ast.parse(raw, filename=name)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = dict(kind=type(node).__name__, line=node.lineno)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != '*':
                    found[alias.asname or alias.name] = dict(kind='import', line=node.lineno,
                        module='.' * node.level + (node.module or ''), symbol=alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    found[target.id] = dict(kind='assignment', line=node.lineno)
    return tree, found


def aliases(tree):
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        for name in names:
            if name not in ['MODEL_REMAPPING', 'MODEL_ARCHITECTURE_REMAPPING']:
                continue
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError('Literal upstream alias dictionary required')
            rows = []
            for key, target in value.items():
                keys = [key] if isinstance(key, str) else list(key) if isinstance(key, tuple) else []
                if not keys or any(type(k) is not str for k in keys) or type(target) is not str:
                    raise ValueError('String model/architecture alias required')
                rows.append(dict(source=keys, target=target))
            result[name] = sorted(rows, key=lambda row: row['source'])
    return result



def selected_path(name, package):
    return (name.startswith(package + '/models/') and name.endswith('.py')
        or name in [package + '/utils.py', 'pyproject.toml', 'setup.py', 'LICENSE', 'README.md'])


def tree_selection(tree, commit, project, revision):
    """Reconstruct every Git tree object, including the commit's root tree."""
    if commit['sha'] != revision or tree['truncated'] is not False:
        raise ValueError('Pinned commit and complete recursive tree required')
    root = commit['commit']['tree']['sha']
    if tree['sha'] != root or not re.fullmatch('[0-9a-f]{40}', root):
        raise ValueError('Commit root tree differs')
    rows = tree['tree']
    if not isinstance(rows, list) or len(rows) > MAX_MEMBERS:
        raise ValueError('Bounded recursive tree required')
    entries = {}
    children = {'': []}
    modes = {'040000': 'tree', '100644': 'blob', '100755': 'blob', '120000': 'blob', '160000': 'commit'}
    for row in rows:
        name = row['path'];path = PurePosixPath(name)
        if (not name or len(name) > 4096 or len(path.parts) > 128 or name.startswith('/')
            or any(p in ['.', '..'] for p in path.parts) or str(path) != name or name in entries
            or row['mode'] not in modes or row['type'] != modes[row['mode']]
            or not re.fullmatch('[0-9a-f]{40}', row['sha'])):
            raise ValueError('Canonical unique Git entries required')
        entries[name] = row
        if row['type'] == 'tree': children[name] = []
    for name, row in entries.items():
        parent = str(PurePosixPath(name).parent)
        parent = '' if parent == '.' else parent
        if parent not in children:
            raise ValueError('Missing parent Git tree')
        children[parent].append(row)
    for name, rows in children.items():
        ordered = sorted(rows, key=lambda row: PurePosixPath(row['path']).name.encode() + (b'/' if row['type'] == 'tree' else b''))
        payload = b''.join((row['mode'].lstrip('0') + ' ').encode() + PurePosixPath(row['path']).name.encode()
            + b'\0' + bytes.fromhex(row['sha']) for row in ordered)
        actual = hashlib.sha1(b'tree ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        if actual != (entries[name]['sha'] if name else root):
            raise ValueError('Recursive Git tree object identity differs')
    package = project.replace('-', '_')
    selected = {name: row for name, row in entries.items() if selected_path(name, package)}
    if any(row['type'] != 'blob' or row['mode'] not in ['100644', '100755']
        or type(row.get('size')) is not int or not 0 <= row['size'] <= MAX_SOURCE for row in selected.values()):
        raise ValueError('Bounded ordinary source blobs required')
    return selected, root


def metadata(path, expected):
    with Path(path).open('rb') as stream: raw = stream.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2 or digest(raw) != expected:
        raise ValueError('Bounded metadata identity differs')
    return json.loads(raw)


def inventory(path, expected, project, revision, tree, commit):
    if project not in ['mlx-lm', 'mlx-vlm'] or not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Explicit project and immutable revision required')
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Literal archive SHA256 required')
    # Consume one admitted byte buffer: pathname replacement after this read
    # cannot substitute a different archive during static analysis.
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_ARCHIVE + 1)
    if len(raw) > MAX_ARCHIVE or digest(raw) != expected:
        raise ValueError('Bounded source archive identity differs')
    selection, root_tree = tree_selection(tree, commit, project, revision)
    # Bound all expanded bytes BEFORE tarfile can consume PAX/GNU headers.
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded_raw = stream.read(MAX_EXPANDED + 1)
    if len(expanded_raw) > MAX_EXPANDED:
        raise ValueError('Decompressed tar stream exceeds bound')
    package = project.replace('-', '_')
    prefix = project + '-' + revision
    modules = {}
    files = {}
    seen = set()
    expanded = 0
    count = 0
    remapping = {}
    with tarfile.open(fileobj=io.BytesIO(expanded_raw), mode='r:') as archive:
        for member in archive:
            count += 1
            path_name = member.name.rstrip('/')
            parts = PurePosixPath(path_name).parts
            if (count > MAX_MEMBERS or not parts or parts[0] != prefix
                or path_name.startswith('/') or any(p in ['.', '..'] for p in parts)
                or str(PurePosixPath(path_name)) != path_name or path_name in seen
                or not (member.isdir() or member.isfile())):
                raise ValueError('Confined unique ordinary archive members required')
            seen.add(path_name)
            if member.isdir():
                continue
            expanded += member.size
            if member.size < 0 or expanded > MAX_EXPANDED:
                raise ValueError('Expanded source archive exceeds bound')
            name = '/'.join(parts[1:])
            model = name.startswith(package + '/models/') and name.endswith('.py')
            selected = selected_path(name, package)
            if not selected:
                continue
            if member.size > MAX_SOURCE:
                raise ValueError('Source member exceeds bound')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('Regular source bytes unavailable')
            source = stream.read(MAX_SOURCE + 1)
            if len(source) != member.size:
                raise ValueError('Truncated source member')
            blob = hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
            if (name not in selection or selection[name]['size'] != len(source) or selection[name]['sha'] != blob
                or member.mode & 0o7777 != int(selection[name]['mode'], 8) & 0o7777):
                raise ValueError('Selected Git source blob/mode differs')
            files[name] = dict(bytes=len(source), sha256=digest(source), git_blob_sha1=blob, mode=selection[name]['mode'])
            if name.endswith('.py'):
                tree, exports = symbols(source, name)
                if name == package + '/utils.py':
                    remapping = aliases(tree)
                if model:
                    relative = name.removeprefix(package + '/models/')
                    # LM imports a root file; VLM generally imports a package's
                    # __init__.py. Nested language/vision helper exports remain
                    # visible as modules, not counted as root entry points.
                    root = '/' not in relative or relative.count('/') == 1 and relative.endswith('/__init__.py')
                    entry = root and 'Model' in exports and relative != '__init__.py'
                    identifier = relative.removesuffix('/__init__.py').removesuffix('.py') if entry else None
                    modules[name] = dict(symbols={k: exports[k] for k in ['Model', 'ModelArgs', 'ModelConfig'] if k in exports},
                        root_entry_point=entry, identifier=identifier)
    if set(files) != set(selection) or not modules or package + '/utils.py' not in files:
        raise ValueError('Complete model tree and loader source required')
    entry_points = {row['identifier']: name for name, row in modules.items() if row['root_entry_point']}
    if len(entry_points) != sum(row['root_entry_point'] for row in modules.values()):
        raise ValueError('Ambiguous root entry point')
    return dict(project=project, revision=revision, archive=dict(bytes=len(raw), sha256=expected),
        root_git_tree_sha1=root_tree, source_files=files, model_modules=modules, root_entry_points=entry_points,
        loader_aliases=remapping, archive_members=count,
        scope='Static explicit exports and literal loader aliases only; no import, inference, fit or complete dynamic-registry claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['lm', 'vlm']:
        parser.add_argument('--' + name + '-archive', type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
        parser.add_argument('--' + name + '-revision', required=True)
        for kind in ['tree','commit']:
            parser.add_argument('--' + name + '-' + kind, type=Path, required=True)
            parser.add_argument('--' + name + '-' + kind + '-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Never overwrite an inventory')
    projects = {name: inventory(getattr(args, name + '_archive'), getattr(args, name + '_sha256'),
        'mlx-' + name, getattr(args, name + '_revision'),
        metadata(getattr(args,name + '_tree'),getattr(args,name + '_tree_sha256')),
        metadata(getattr(args,name + '_commit'),getattr(args,name + '_commit_sha256'))) for name in ['lm', 'vlm']}
    result = dict(format='toklet-upstream-mlx-architecture-inventory-v1', projects=projects,
        native_coverage='No Toklet support is conferred by this reference inventory; consult separately qualified engine evidence',
        generator_sha256=digest(Path(__file__).read_bytes()))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
