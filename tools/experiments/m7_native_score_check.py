"""Bounded Mac score-worker checks and unsigned Apple harness builds.

Requires an exact reviewed package receipt with the additive score export.
No signing, physical-device launch or full-quality acceptance occurs here.
"""
import json
import math
from pathlib import Path
import sys
import m7_apple_package as package
import m7_apple_integration as integration
import m7_swift_harness_build as builder
ROOT=package.ROOT

STATES=[('unloaded-reject',1,0,0,0,0),('load',0,0,0,0,0),('empty-reject',1,0,0,0,0),
        ('commit',0,4,4,0,4),('target',0,4,4,0,4),('repeat',0,4,4,0,4),
        ('alternative',0,4,4,0,4),('range-reject',1,4,4,0,4),('recover',0,4,4,0,4),
        ('teacher-append',0,5,5,0,5),('after-append',0,5,5,0,5),('pending',0,6,5,1,6),
        ('pending-reject',1,6,5,1,6),('suspend',0,0,0,0,6),('suspend-reject',1,0,0,0,6),
        ('replay',0,6,6,0,6),('after-replay',0,6,6,0,6),('reset',0,0,0,0,0),
        ('reset-reject',1,0,0,0,0),('unload',0,0,0,0,0)]
SCORES={'target':(47,4),'repeat':(47,4),'alternative':(3,4),'recover':(47,4),'after-append':(47,5),'after-replay':(47,6)}


def validate_output(text,backend):
    data=json.loads(text)
    if (data['passed'] is not True or data['backend']!=backend or data['closed'] is not True
        or data['released'] is not True or data['physicalDeviceQualified'] is not False
        or data['publicationRaces']!=['interrupt-unload','close']):
        raise ValueError('Score worker outcome differs')
    rows=data['checks']
    observed=[tuple(row[k] for k in ('name','status','history','evaluated','pending','saved')) for row in rows]
    if observed!=STATES or any(type(row[k]) is not int for row in rows for k in ('status','history','evaluated','pending','saved')):
        raise ValueError('Exact score worker state sequence required')
    by_name={row['name']:row for row in rows}
    for row in rows:
        score=row.get('score');expected=SCORES.get(row['name'])
        if expected is None:
            if score is not None:raise ValueError('Stale score published')
            continue
        if score is None or (score['target'],score['evaluated'],score['greedy'],score['vocabulary'])!=(*expected,47,64):
            raise ValueError('Score identity differs')
        if any(type(score[k]) is not int for k in ('target','evaluated','greedy','vocabulary')):
            raise ValueError('Noninteger score identity')
        for key in ('targetLogit','greedyLogit','negativeLogLikelihood'):
            if type(score[key]) not in (int,float) or not math.isfinite(score[key]):raise ValueError('Nonfinite score')
        if score['greedyLogit']<score['targetLogit'] or score['negativeLogLikelihood']<=0:
            raise ValueError('Invalid score value')
    first=by_name['target']['score'];alternative=by_name['alternative']['score']
    if first!=by_name['repeat']['score'] or first!=by_name['recover']['score'] or first['targetLogit']!=first['greedyLogit']:
        raise ValueError('Read-only score changed')
    if abs((alternative['negativeLogLikelihood']-first['negativeLogLikelihood'])-(first['targetLogit']-alternative['targetLogit']))>=1e-12:
        raise ValueError('Alternative loss identity differs')
    if backend=='cpu' and abs(first['negativeLogLikelihood']-1.6877391136919981)>=1e-5:
        raise ValueError('Independent tiny CPU loss differs')
    return data


def validate_legacy(text,backend):
    data=json.loads(text)
    expected=[('load',0,0,0,0,0),('prompt',6,5,1,6,1),('continue',8,7,1,8,1),('suspend',0,0,0,8,0),
              ('replay-cancelled',0,0,0,8,3),('replay',8,8,0,8,1),('after-replay',9,8,1,9,1),
              ('reset',0,0,0,0,0),('cancel',5,4,1,5,3),('resume',6,5,1,6,1)]
    if ([tuple(r[k] for k in ('name','history','evaluated','pending','saved','reason')) for r in data['states']]!=expected
        or data['passed'] is not True or data['backend']!=backend or data['tokens']!=[47]*4
        or data['closed'] is not True or data['released'] is not True or data['physicalDeviceQualified'] is not False):
        raise ValueError('Legacy worker state/oracle differs')
    return data


def main(native,expected,output):
    native=native.resolve();output=output.resolve()
    if not native.is_relative_to(ROOT/'artifacts') or not output.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Repository artifacts required')
    output.mkdir();logs=output/'logs';logs.mkdir();temporary=output/'tmp';temporary.mkdir()
    pins={};commands=[];checked={};primary=None;endpoints=[];app_sha=None;env={}
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    def merge(items):integration.merge(pins,items)
    def guard():
        for name,digest in pins.items():
            if package.sha(ROOT/name)!=digest:raise ValueError('Changed score-check input/output: '+name)
    def execute(command,label,timeout=180):
        guard();row=package.owned(command,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row)
        if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(label+': '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
        for field in ('stdout','stderr'):merge({row[field]:package.sha(ROOT/row[field])})
        guard()
        return ROOT/row['stdout']
    try:
        env,_=package.build_environment();env['TMPDIR']=str(temporary)
        merge(integration.admit_build(native,expected));merge(integration.FIXTURE)
        sources=[ROOT/'apple/Harness/NativeWorker.swift',ROOT/'apple/tests/swift/ScoreMain.swift',
                 ROOT/'apple/tests/swift/WorkerChecks.swift',ROOT/'apple/tests/swift/main.swift']
        helpers=[Path(module.__file__).resolve() for module in (package,integration,builder)]
        helpers.extend([Path(__file__).resolve(),Path(__file__).with_name('test_m7_native_score_check.py')])
        merge({str(p.relative_to(ROOT)):package.sha(p) for p in sources+helpers})
        guard()
        # Receipt bytes are hashed and parsed together; no second unbound read.
        with (native/'build.json').open('rb') as stream:build_raw=stream.read(16*1024**2+1)
        if len(build_raw)>16*1024**2 or package.hashlib.sha256(build_raw).hexdigest()!=expected:raise ValueError('Package receipt differs')
        build=json.loads(build_raw)
        if any('toklet_session_score' not in row['exported_c_symbols'] for row in build['slices']):raise ValueError('Score export required')
        execute(['xcodebuild','-version'],'xcode-version')
        sdk=package.small(execute(['xcrun','--sdk','macosx','--show-sdk-path'],'sdk-path')).strip()
        if sdk!='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk':raise ValueError('Unadmitted Mac SDK')
        execute(['xcrun','--sdk','macosx','swiftc','--version'],'swift-version')
        archive=native/'Toklet.xcframework/macos-arm64/libtoklet.a'
        flags=['xcrun','--sdk','macosx','swiftc','-sdk',sdk,'-module-cache-path',str(output/'ModuleCache'),
               '-swift-version','5','-strict-concurrency=complete','-warnings-as-errors','-target','arm64-apple-macos15.0','-O','-I',str(ROOT/'apple/include')]
        links=[str(archive),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security',
               '-lobjc','-liconv','-lresolv','-lc++']
        for label,files in [('score',[sources[0],sources[1]]),('legacy',[sources[0],sources[2],sources[3]])]:
            binary=output/label
            execute(flags+[str(p) for p in files]+links+['-o',str(binary)],label+'-compile')
            merge({str(binary.relative_to(ROOT)):package.sha(binary)})
            for backend,extra in [('cpu',[]),('metal',['metal'])]:
                stdout=execute([str(binary),str(ROOT/'artifacts/m7-native-tiny-v2'),*extra],label+'-'+backend)
                parser=validate_output if label=='score' else validate_legacy
                checked[label+'-'+backend]=parser(package.small(stdout),backend)
        apps=output/'apple-build'
        execute([sys.executable,str(Path(builder.__file__).resolve()),str(native),expected,str(apps)],'apple-harness-build',1800)
        with (apps/'build.json').open('rb') as stream:raw=stream.read(16*1024**2+1)
        if len(raw)>16*1024**2:raise ValueError('Oversized app receipt')
        app_sha=package.hashlib.sha256(raw).hexdigest();app=json.loads(raw)
        if not app['passed'] or app['primary_error'] or app['endpoint_errors'] or len(app['apps'])!=2:raise ValueError('Apple harness build failed')
        merge({str((apps/'build.json').relative_to(ROOT)):app_sha});merge(app['bindings']);merge(app['bundle_pins'])
        builder.verify_library_link(app['library_link'])
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        package.signal.signal(package.signal.SIGTERM,previous)
        try:guard()
        except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
    passed=primary is None and not endpoints and len(commands)==10 and len(checked)==4 and app_sha is not None
    record=dict(format='toklet-m7-native-score-check-v1',passed=passed,collection_root=str(ROOT),bindings=pins,
                commands=commands,controlled_environment=env,checked=checked,app_build_sha256=app_sha,
                primary_error=primary,endpoint_errors=endpoints,full_quality_qualified=False,physical_device_qualified=False,m7_complete=False)
    with (output/'checks.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])))
