"""Concrete cold data-parser closure; shared by two quality-input replayers."""
from pathlib import Path
import sysconfig

def sources(scripts):
    site=Path(sysconfig.get_path('purelib')).resolve()
    selected=[Path(__file__).resolve(),*map(Path,scripts)]
    for package in ('pyarrow','numpy'):
        folder=site/package
        if not folder.is_dir():raise ValueError('Quality preparation package missing: '+package)
        selected.extend(q for q in folder.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
    for pattern in ('pyarrow*.dist-info','numpy*.dist-info','numpy*.libs','pyarrow*.libs'):
        for folder in site.glob(pattern):selected.extend(q for q in folder.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
    return sorted({q.resolve() for q in selected})


def admit(path,expected,selected):
    """Add explicit cold/loader checks while preserving legacy text admission."""
    import sys
    from importlib.machinery import SourceFileLoader,ExtensionFileLoader,PathFinder
    from importlib.util import find_spec
    import text_reference_runtime as text
    if any(name==root or name.startswith(root+'.') for name in sys.modules for root in ('numpy','pyarrow')):
        raise ValueError('quality parser package already imported')
    site=Path(sysconfig.get_path('purelib')).resolve();inventory={str(q) for q in text.paths(selected)};imports={}
    for name in ('numpy','pyarrow'):
        folder=site/name;spec=find_spec(name)
        if spec is None or spec.origin!=str(folder/'__init__.py') or not isinstance(spec.loader,SourceFileLoader) or list(spec.submodule_search_locations or [])!=[str(folder)] or spec.origin not in inventory:
            raise ValueError('quality parser root loader/origin differs')
        imports[name]=spec.origin
    native=PathFinder.find_spec('pyarrow.lib',[str(site/'pyarrow')])
    core=PathFinder.find_spec('numpy._core',[str(site/'numpy')])
    if core is None or not isinstance(core.loader,SourceFileLoader) or core.origin not in inventory:
        raise ValueError('NumPy core loader/origin differs')
    array=PathFinder.find_spec('numpy._core._multiarray_umath',list(core.submodule_search_locations or []))
    for name,spec in [('pyarrow.lib',native),('numpy._core._multiarray_umath',array)]:
        if spec is None or not isinstance(spec.loader,ExtensionFileLoader) or spec.origin not in inventory:
            raise ValueError('quality parser native loader/origin differs')
        imports[name]=spec.origin
    admission,guard=text.admit(path,expected,selected)
    admission['extra_imports']=imports
    def endpoint():
        for name,origin in imports.items():
            actual=getattr(sys.modules.get(name),'__file__',None)
            if actual is None or str(Path(actual).resolve())!=origin:raise ValueError('actual quality parser origin differs: '+name)
        return guard()
    return admission,endpoint
