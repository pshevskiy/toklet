"""Conservative observed filesystem admission for owned phone transfers.

This reserves no disk space. All copy bytes plus a fixed reserve must fit a
recent same-device/app observation. Both host clocks and a boot identity bind
the age; the measurement start precedes the device's filesystem read.
"""
import json
import uuid
import m7_apple_package as package
import m7_apple_integration as integration
import m7_physical_integration as physical

ROOT=package.ROOT
RESERVE=2*1024**3
MAX_AGE_NS=300*10**9


def unsigned(value):return type(value) is int and 0<=value<2**64
def digest(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def boot_token(text):
    value=text.strip()
    try:parsed=uuid.UUID(value)
    except (ValueError,AttributeError) as error:raise ValueError('Host boot identity unavailable') from error
    if len(value)!=36 or str(parsed)!=value.lower() or parsed.int==0:raise ValueError('Host boot identity unavailable')
    # The raw kernel boot UUID remains private; only its hash binding is
    # public. This is a boot identity, never a phone or hardware identifier.
    return package.hashlib.sha256(('darwin-kern.bootsessionuuid-v1:'+str(parsed)).encode()).hexdigest()


def verify(pins):
    for name,pin in pins.items():
        if package.sha(ROOT/name)!=pin:raise ValueError('Changed storage proof input: '+name)


def size_and_hash(path):
    # The admitted size must describe the very bytes whose SHA is checked.
    # A separate stat/open pair could observe a short-lived smaller file.
    count=0;hasher=package.hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(256*1024):
            if count>2**64-1-len(data):raise ValueError('Storage file size overflows')
            count+=len(data);hasher.update(data)
    return count,hasher.hexdigest()


def inventory(base,config,name,job_sha):
    wanted=dict(config['files']);wanted[name]=job_sha
    actual=set()
    for path in base.rglob('*'):
        if path.is_symlink():raise ValueError('Transfer symlink rejected')
        if path.is_file():actual.add(str(path.relative_to(base)))
        elif not path.is_dir():raise ValueError('Nonregular transfer entry')
    if actual!=set(wanted):raise ValueError('Exact storage payload inventory differs')
    result={};total=0
    for target,pin in sorted(wanted.items()):
        path=base/target;size,observed=size_and_hash(path)
        if not unsigned(size) or size==0 or total>2**64-1-size or observed!=pin:
            raise ValueError('Storage payload size/hash/overflow differs')
        total+=size;result[target]=dict(source=str(path.relative_to(ROOT)),sha256=pin,bytes=size)
    if total>2**64-1-RESERVE:raise ValueError('Storage reserve addition overflows')
    return result,total


def admit_proof(folder,expected,context,check_console):
    p=folder/'measurement.json';r=integration.metadata(p,expected)
    if (r['format']!='toklet-m7-physical-measurement-v1' or r['passed'] is not True
        or r['config']!=context['tiny_config'] or r['prepared_receipt']!=context['prepared_receipt']
        or r['signed_bundle']!=context['signed_bundle'] or r['primary_error'] is not None
        or r['endpoint_errors'] or r['cleanup_errors'] or len(r['remote_cleanup'])!=2
        or r['physical_device_qualified'] is not False or r['m7_complete'] is not False
        or any(not c['absent'] or c['errors'] for c in r['remote_cleanup'])
        or r['bindings'].get(context['selection_path'])!=context['selection_sha']
        or any(r['bindings'].get(n)!=h for n,h in context['required_bindings'].items())):
        raise ValueError('Same-device signed short observation required')
    start=r.get('measurement_host_start');boot=r.get('host_boot_token')
    if (type(start) is not dict or set(start)!={'wallUnixNanoseconds','monotonicNanoseconds'}
        or not all(unsigned(v) and v>0 for v in start.values()) or not digest(boot)):
        raise ValueError('Conservative host start and boot identity required')
    required=[folder/'private-logs'/n for n in ('measurement.stdout','measurement.stderr','launch.json','host-boot.stdout')]
    if any(str(p.relative_to(ROOT)) not in r['private_evidence_pins'] for p in required):
        raise ValueError('Storage observation raw evidence is unbound')
    verify(r['bindings']);verify(r['private_evidence_pins'])
    if boot_token(package.small(required[3]))!=boot:raise ValueError('Host boot proof differs')
    checked=check_console(package.small(required[0])+'\n'+package.small(required[1]),r['config'],r['signed_bundle'])
    checked['launch']=physical.check_launch(json.loads(package.small(required[2])),context['remote'],context['selected'])
    if checked!=r['checked']:raise ValueError('Storage observation replay differs')
    pins=dict(r['bindings']);integration.merge(pins,r['private_evidence_pins'])
    integration.merge(pins,{str(p.relative_to(ROOT)):expected})
    return r,pins


def decision(proof,folder,proof_sha,payload,total):
    free=proof['checked']['result']['filesystemBefore']['freeBytes']
    if not unsigned(total) or total>2**64-1-RESERVE or free<total+RESERVE:
        raise ValueError('Observed free bytes do not cover payload plus reserve')
    return dict(format='toklet-m7-storage-admission-v1',proof=dict(path=str((folder/'measurement.json').relative_to(ROOT)),sha256=proof_sha),
        payload_inventory=payload,payload_bytes=total,reserve_bytes=RESERVE,required_bytes=total+RESERVE,
        filesystem_observation=proof['checked']['result']['filesystemBefore'],measurement_host_start=proof['measurement_host_start'],
        host_boot_token=proof['host_boot_token'],maximum_age_nanoseconds=MAX_AGE_NS,checks=[])


def fresh(admission,boot,wall,monotonic,label):
    if not digest(boot) or boot!=admission['host_boot_token']:raise ValueError('Storage observation belongs to another host boot')
    start=admission['measurement_host_start']
    if not unsigned(wall) or not unsigned(monotonic):raise ValueError('Invalid storage admission clocks')
    wall_age=wall-start['wallUnixNanoseconds'];mono_age=monotonic-start['monotonicNanoseconds']
    if not (0<=wall_age<MAX_AGE_NS and 0<=mono_age<MAX_AGE_NS):raise ValueError('Storage observation stale or host clock regressed')
    return dict(label=label,wallUnixNanoseconds=wall,monotonicNanoseconds=monotonic,wall_age_nanoseconds=wall_age,monotonic_age_nanoseconds=mono_age)


def admit_ready(record,base,config,name,job_sha,context,check_console):
    saved=record.get('storage_admission')
    if type(saved) is not dict:raise ValueError('Real-model ready receipt lacks original storage admission')
    proof=saved['proof'];path=(ROOT/proof['path']).resolve()
    if path.name!='measurement.json' or not path.is_relative_to(ROOT/'artifacts'):raise ValueError('Invalid original storage proof path')
    observed,pins=admit_proof(path.parent,proof['sha256'],context,check_console)
    payload,total=inventory(base,config,name,job_sha)
    expected=decision(observed,path.parent,proof['sha256'],payload,total)
    labels=['copy-Model']+(['copy-Packed'] if config['packedFolder'] else [])+['copy-job']
    checks=saved.get('checks')
    if type(checks) is not list or len(checks)!=len(labels):raise ValueError('Original storage checks incomplete')
    expected['checks']=[fresh(expected,observed['host_boot_token'],c['wallUnixNanoseconds'],c['monotonicNanoseconds'],label)
                        for c,label in zip(checks,labels)]
    if expected!=saved:raise ValueError('Original storage decision/inventory differs')
    return saved,pins
