"""Sealed tiny quality/admission checks plus measurement regressions/SDK builds."""
import json
import math
from pathlib import Path
import struct
import sys
import m7_apple_package as package
import m7_apple_integration as integration
import m7_native_teacher_check as teacher
import m7_native_measurement_check as measurement
import m7_swift_harness_build as builder
ROOT=package.ROOT
FILES={Path(name).name:sha for name,sha in integration.FIXTURE.items()}
EXPECTED_FILES={'Model/'+name:sha for name,sha in FILES.items()}
TOKEN_SHA=package.hashlib.sha256(struct.pack('<4I',1,7,13,3)).hexdigest()
CASES=['cpu-complete','metal-complete',*('reject-'+key for key in ['version','overflow-targets','overflow-block','capacity','chunk','budget','corpus-path','corpus-in-model','corpus-hash','inventory']),
       'reject-extra-fields','reject-config-sha','reject-corpus-size','reject-corpus-symlink','reject-output','reject-model-endpoint','reject-corpus-endpoint']


def footprint(value):
    a,b=value['physicalBytes'],value['lifetimePeakPhysicalBytes']
    if type(a) is not int or type(b) is not int or not 0<a<=b<=192*1024**2:raise ValueError('Quality process peak exceeded')


def validate_output(text,oracle):
    if len(oracle)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in oracle):raise ValueError('Complete independent quality oracle required')
    markers=['QUALITY_CHECK_SAMPLE_JSON=','QUALITY_CHECK_RESULT_JSON=','QUALITY_CHECK_FAILURE_JSON=','QUALITY_CHECKS_JSON=']
    observed=[]
    for line in text.splitlines():
        if not line.strip():continue
        matches=[m for m in markers if line.startswith(m)]
        if len(matches)!=1:raise ValueError('Unexpected quality output/error marker')
        observed.append(matches[0])
    expected=[markers[0]]*3+[markers[1]]+[markers[0]]*3+[markers[1]]+[markers[2]]*5+[markers[3]]
    if observed!=expected:raise ValueError('Quality stream marker sequence differs')
    def rows(marker):return [json.loads(line[len(marker):]) for line in text.splitlines() if line.startswith(marker)]
    samples=rows('QUALITY_CHECK_SAMPLE_JSON=');results=rows('QUALITY_CHECK_RESULT_JSON=')
    failures=rows('QUALITY_CHECK_FAILURE_JSON=');summaries=rows('QUALITY_CHECKS_JSON=')
    if len(samples)!=6 or len(results)!=2 or len(failures)!=5 or len(summaries)!=1:raise ValueError('Incomplete native quality checks')
    for backend,result,group in zip(('cpu','metal'),results,(samples[:3],samples[3:])):
        if (result['backend']!=backend or result['passed'] is not True or result.get('error') is not None
            or result['cleanupErrors'] or result['endpointErrors'] or result['closed'] is not True or result['released'] is not True
            or result['fullQualityQualified'] is not False or result['physicalDeviceQualified'] is not False or result['m7Complete'] is not False
            or result['interruptionCount']!=0 or result['samplesCompleted']!=3 or result['filesBefore']!=EXPECTED_FILES
            or result['filesAfter']!=EXPECTED_FILES or result.get('tokensBeforeSHA256')!=TOKEN_SHA or result.get('tokensAfterSHA256')!=TOKEN_SHA):
            raise ValueError('Quality result/proof/cleanup differs')
        if [c['phase'] for c in result['checkpoints']]!=['before-artifact-proof','after-artifact-proof','loaded','block-0','unloaded','closed','after-final-artifact-proof']:
            raise ValueError('Quality checkpoint phases differ')
        elapsed=result['loadAndSessionSeconds']
        if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid quality load bracket')
        if any(type(result[key]) is not int for key in ('interruptionCount','samplesCompleted')):raise ValueError('Quality counts must be integers')
        for checkpoint in result['checkpoints']:footprint(checkpoint['footprint'])
        losses=[]
        for index,sample in enumerate(group):
            footprint(sample['footprint']);row=sample['target'];score=row['score']
            if tuple(row[k] for k in ('targetIndex','block','position','committedHistory'))!=(index,0,index+1,index+2):raise ValueError('Quality row order/prefix differs')
            if any(type(row[k]) is not int for k in ('targetIndex','block','position','committedHistory')):raise ValueError('Quality row counts must be integers')
            if (score['target']!=(7,13,3)[index] or score['evaluated']!=index+1 or score['vocabulary']!=64
                or any(type(score[k]) is not int for k in ('target','evaluated','vocabulary','greedy')) or not 0<=score['greedy']<64):raise ValueError('Quality score identity differs')
            for key in ('targetLogit','greedyLogit','negativeLogLikelihood'):
                if type(score[key]) not in (int,float) or not math.isfinite(score[key]):raise ValueError('Nonfinite quality score')
            if score['greedyLogit']<score['targetLogit'] or score['negativeLogLikelihood']<0:raise ValueError('Invalid quality loss')
            env=sample['environment']
            if env.get('applicationState') is not None or type(env['thermalState']) is not int or env['thermalState'] not in range(4) or type(env['lowPowerMode']) is not bool:
                raise ValueError('Mac quality environment differs')
            losses.append(score['negativeLogLikelihood'])
        summary=result['summary']
        if (summary['targets'],summary['blocks'],summary['blockSize'],summary['policy'])!=(3,1,3,teacher.POLICY):raise ValueError('Quality summary policy differs')
        if any(type(summary[key]) is not int for key in ('targets','blocks','blockSize')):raise ValueError('Quality summary counts must be integers')
        for key in ('negativeLogLikelihoodSum','meanNegativeLogLikelihood'):
            if type(summary[key]) not in (int,float) or not math.isfinite(summary[key]):raise ValueError('Invalid quality aggregate')
        if abs(summary['negativeLogLikelihoodSum']-math.fsum(losses))>1e-12 or abs(summary['meanNegativeLogLikelihood']-math.fsum(losses)/3)>1e-12:raise ValueError('Quality aggregate differs')
        if backend=='cpu' and (len(oracle)!=3 or any(abs(x-y)>=1e-5 for x,y in zip(losses,oracle))):raise ValueError('Independent quality CPU loss differs')
    labels=['overflow-targets','overflow-block','output','model-endpoint','corpus-endpoint']
    if [r['label'] for r in failures]!=labels:raise ValueError('Exact quality failure inventory required')
    for row in failures:
        result=row['result'];label=row['label']
        if result['passed'] is not False or result['released'] is not True or result['fullQualityQualified'] is not False or result['physicalDeviceQualified'] is not False or result['m7Complete'] is not False:
            raise ValueError('Failure qualified or leaked owner')
        if label.startswith('overflow'):
            if (not result.get('error') or result['samplesCompleted']!=0 or result.get('summary') is not None or result['closed'] is not False
                    or result['filesBefore'] or result['filesAfter'] or result.get('tokensBeforeSHA256') is not None
                    or result.get('tokensAfterSHA256') is not None or result['loadAndSessionSeconds']!=0
                    or [c['phase'] for c in result['checkpoints']]!=['closed']):raise ValueError('Overflow policy issued native work')
        else:
            if result['closed'] is not True or result['cleanupErrors']:raise ValueError('Failed quality cleanup incomplete')
            if label=='output':
                if (not result.get('error') or result['samplesCompleted']!=1 or result.get('summary') is not None or result['endpointErrors']
                        or result['filesBefore']!=EXPECTED_FILES or result['filesAfter']!=EXPECTED_FILES
                        or result.get('tokensBeforeSHA256')!=TOKEN_SHA or result.get('tokensAfterSHA256')!=TOKEN_SHA):raise ValueError('Failed sink completed summary')
            else:
                if result.get('error') is not None or result['samplesCompleted']!=3 or not result['endpointErrors'] or result['summary']['targets']!=3:raise ValueError('Endpoint mutation was accepted')
                if (result['filesBefore']!=EXPECTED_FILES or result.get('tokensBeforeSHA256')!=TOKEN_SHA
                    or result.get('tokensAfterSHA256') is not None
                    or result['filesAfter']!=({} if label=='model-endpoint' else EXPECTED_FILES)):
                    raise ValueError('Failure endpoint proof correspondence differs')
    if summaries[0] != dict(passed=True,cases=CASES,physicalDeviceQualified=False):raise ValueError('Quality admission cases incomplete')
    return dict(results=results,sample_count=6,failures=failures,summary=summaries[0])


def main(native,expected,output):
    native=native.resolve();output=output.resolve()
    if not native.is_relative_to(ROOT/'artifacts') or not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Repository artifacts required')
    output.mkdir();logs=output/'logs';logs.mkdir();(output/'tmp').mkdir()
    pins={};commands=[];checked=None;primary=None;endpoints=[];env={};nested={};links=[]
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    def merge(items):integration.merge(pins,items)
    def guard():
        for link in links:builder.verify_library_link(link)
        for name,digest in pins.items():
            if package.sha(ROOT/name)!=digest:raise ValueError('Changed quality-check input/output: '+name)
    def execute(args,label,timeout=180):
        guard();row=package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row)
        if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(label+': '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
        for key in ('stdout','stderr'):merge({row[key]:package.sha(ROOT/row[key])})
        guard();return ROOT/row['stdout']
    def admit_nested(folder):
        path=folder/('checks.json' if (folder/'checks.json').exists() else 'build.json')
        with path.open('rb') as stream:raw=stream.read(16*1024**2+1)
        if len(raw)>16*1024**2:raise ValueError('Oversized nested receipt')
        digest=package.hashlib.sha256(raw).hexdigest();record=json.loads(raw)
        if not record['passed'] or record['primary_error'] or record['endpoint_errors']:raise ValueError('Nested quality regression/build failed')
        merge({str(path.relative_to(ROOT)):digest});merge(record['bindings'])
        if 'bundle_pins' in record:merge(record['bundle_pins'])
        if 'library_link' in record:links.append(record['library_link'])
        for row in record['commands']:
            if row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:raise ValueError('Nested owned command failed')
            for key in ('stdout','stderr'):merge({row[key]:package.sha(ROOT/row[key])})
        nested[str(path.relative_to(ROOT))]=digest
        return record
    try:
        env,_=package.build_environment();env['TMPDIR']=str(output/'tmp')
        merge(integration.admit_build(native,expected));merge(integration.FIXTURE);merge(teacher.ORACLE)
        sources=[ROOT/'apple/Harness'/name for name in ['NativeWorker.swift','NativeMeasurement.swift','NativeTeacherForcing.swift','NativeQuality.swift']]
        sources.append(ROOT/'apple/tests/swift/QualityMain.swift')
        helpers=[Path(m.__file__).resolve() for m in (package,integration,teacher,measurement,builder)]
        helpers += [Path(__file__).resolve(),Path(__file__).with_name('test_m7_native_quality_check.py'),ROOT/'apple/Harness/HarnessApp.swift',ROOT/'apple/TokletHarness.xcodeproj/project.pbxproj']
        merge({str(p.relative_to(ROOT)):package.sha(p) for p in sources+helpers});guard()
        with (ROOT/'tests/fixtures/tiny-v1/cases.prompt.logits.f32le').open('rb') as stream:raw=stream.read(1025)
        oracle=teacher.independent_losses(raw)
        execute(['xcodebuild','-version'],'xcode-version')
        execute(['xcrun','--sdk','macosx','swiftc','--version'],'swift-version')
        sdk=package.small(execute(['xcrun','--sdk','macosx','--show-sdk-path'],'sdk-path')).strip()
        if sdk!='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk':raise ValueError('Unadmitted Mac SDK')
        binary=output/'quality'
        execute(['xcrun','--sdk','macosx','swiftc','-sdk',sdk,'-module-cache-path',str(output/'ModuleCache'),'-swift-version','5',
                 '-strict-concurrency=complete','-warnings-as-errors','-target','arm64-apple-macos15.0','-O','-I',str(ROOT/'apple/include'),
                 *[str(p) for p in sources],str(native/'Toklet.xcframework/macos-arm64/libtoklet.a'),'-framework','Metal',
                 '-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security','-lobjc','-liconv','-lresolv','-lc++','-o',str(binary)],'compile')
        merge({str(binary.relative_to(ROOT)):package.sha(binary)})
        stdout=execute([str(binary),str(ROOT/'artifacts/m7-native-tiny-v2'),str(output/'cases')],'quality-checks',300)
        checked=validate_output(package.small(stdout),oracle)
        for label,module in [('measurement-regression',measurement),('apple-build',builder)]:
            destination=output/label
            execute([sys.executable,str(Path(module.__file__).resolve()),str(native),expected,str(destination)],label,1800)
            record=admit_nested(destination)
            if label=='measurement-regression':
                if record['checked']!=measurement.validate_output(package.small(destination/'logs/checks.stdout')):raise ValueError('Measurement regression replay differs')
            else:builder.verify_library_link(record['library_link'])
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        package.signal.signal(package.signal.SIGTERM,previous)
        try:guard()
        except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
    passed=primary is None and not endpoints and len(commands)==7 and len(nested)==2 and checked is not None
    record=dict(format='toklet-m7-native-quality-check-v1',passed=passed,collection_root=str(ROOT),bindings=pins,commands=commands,
                checked=checked,nested_receipts=nested,controlled_environment=env,primary_error=primary,endpoint_errors=endpoints,
                full_quality_qualified=False,physical_device_qualified=False,m7_complete=False)
    with (output/'checks.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])))
