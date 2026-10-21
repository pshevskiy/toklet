"""Compile and check the cold teacher-forcing coordinator; no device actions."""
import json
import math
from pathlib import Path
import struct
import sys
import m7_apple_package as package
import m7_apple_integration as integration
ROOT=package.ROOT
POLICY='independent_blocks_of_up_to_1025_tokens_one_token_overlap_score_every_target_once'
SIMULATED=['frozen-10000-target-policy','partial-block','append-failure-prefix','score-prefix-rejection',
           'cancelled-prompt','output-failure-prefix','preflight-bounds']
ORACLE={'tests/fixtures/tiny-v1/manifest.json':'1d40cec4f24ece5fdc7b727d3e96f41237621120908833ee6c7df83352acfdd4',
        'tests/fixtures/tiny-v1/cases.prompt.logits.f32le':'45b1def79d59fd9b37fa2b8504af2b5b2eea72d41a2c44187f9cd68e46b95c23'}


def independent_losses(raw):
    if len(raw)!=1024 or package.hashlib.sha256(raw).hexdigest()!=ORACLE['tests/fixtures/tiny-v1/cases.prompt.logits.f32le']:
        raise ValueError('Independent tiny logits differ')
    values=struct.unpack('<256f',raw);losses=[]
    for index,target in enumerate((7,13,3)):
        row=values[index*64:(index+1)*64];maximum=max(row)
        losses.append((maximum-row[target])+math.log(math.fsum(math.exp(value-maximum) for value in row)))
    return losses


def validate_output(text,backend,oracle):
    if len(oracle)!=3 or any(not math.isfinite(x) or x<0 for x in oracle):raise ValueError('Complete independent oracle required')
    data=json.loads(text)
    if (data['passed'] is not True or data['backend']!=backend or data['simulated']!=SIMULATED
        or data['closed'] is not True or data['released'] is not True or data['physicalDeviceQualified'] is not False):
        raise ValueError('Teacher coordinator checks incomplete')
    rows=data['rows'];summary=data['summary']
    if len(rows)!=3 or summary['targets']!=3 or summary['blocks']!=1 or summary['blockSize']!=3 or summary['policy']!=POLICY:
        raise ValueError('Teacher policy/coverage differs')
    losses=[]
    for index,row in enumerate(rows):
        expected=(index,0,index+1,index+2)
        if tuple(row[k] for k in ('targetIndex','block','position','committedHistory'))!=expected or any(type(row[k]) is not int for k in ('targetIndex','block','position','committedHistory')):
            raise ValueError('Teacher row order/commit differs')
        score=row['score']
        if (score['target']!=(7,13,3)[index] or score['evaluated']!=index+1 or score['vocabulary']!=64
            or any(type(score[k]) is not int for k in ('target','evaluated','vocabulary','greedy'))
            or not 0<=score['greedy']<64):raise ValueError('Teacher score identity differs')
        for k in ('targetLogit','greedyLogit','negativeLogLikelihood'):
            if type(score[k]) not in (int,float) or not math.isfinite(score[k]):raise ValueError('Nonfinite teacher score')
        if score['greedyLogit']<score['targetLogit'] or score['negativeLogLikelihood']<0:raise ValueError('Invalid teacher score')
        losses.append(score['negativeLogLikelihood'])
    for key in ('targets','blocks','blockSize'):
        if type(summary[key]) is not int:raise ValueError('Teacher summary counts must be integers')
    for key in ('negativeLogLikelihoodSum','meanNegativeLogLikelihood'):
        if type(summary[key]) not in (int,float) or not math.isfinite(summary[key]):raise ValueError('Invalid summary loss')
    if abs(summary['negativeLogLikelihoodSum']-math.fsum(losses))>1e-12 or abs(summary['meanNegativeLogLikelihood']-math.fsum(losses)/3)>1e-12:
        raise ValueError('Teacher loss aggregation differs')
    if backend=='cpu' and any(abs(value-reference)>=1e-5 for value,reference in zip(losses,oracle)):
        raise ValueError('Independent teacher CPU loss differs')
    return data


def main(native,expected,output):
    native=native.resolve();output=output.resolve()
    if not native.is_relative_to(ROOT/'artifacts') or not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Repository artifacts required')
    output.mkdir();logs=output/'logs';logs.mkdir();(output/'tmp').mkdir()
    pins={};commands=[];checked={};primary=None;endpoints=[];env={};oracle=[]
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    def merge(items):integration.merge(pins,items)
    def guard():
        for n,h in pins.items():
            if package.sha(ROOT/n)!=h:raise ValueError('Changed teacher-check input/output: '+n)
    def execute(args,label):
        guard();row=package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,180);commands.append(row)
        if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(label+': '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
        for key in ('stdout','stderr'):merge({row[key]:package.sha(ROOT/row[key])})
        guard();return ROOT/row['stdout']
    try:
        env,_=package.build_environment();env['TMPDIR']=str(output/'tmp')
        merge(integration.admit_build(native,expected));merge(integration.FIXTURE);merge(ORACLE)
        sources=[ROOT/'apple/Harness/NativeWorker.swift',ROOT/'apple/Harness/NativeTeacherForcing.swift',ROOT/'apple/tests/swift/TeacherForcingMain.swift']
        helpers=[Path(module.__file__).resolve() for module in (package,integration)]
        helpers.extend([Path(__file__).resolve(),Path(__file__).with_name('test_m7_native_teacher_check.py')])
        merge({str(p.relative_to(ROOT)):package.sha(p) for p in sources+helpers});guard()
        with (ROOT/'tests/fixtures/tiny-v1/cases.prompt.logits.f32le').open('rb') as stream:raw=stream.read(1025)
        oracle=independent_losses(raw)
        execute(['xcodebuild','-version'],'xcode-version')
        execute(['xcrun','--sdk','macosx','swiftc','--version'],'swift-version')
        sdk=package.small(execute(['xcrun','--sdk','macosx','--show-sdk-path'],'mac-sdk')).strip()
        if sdk!='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk':raise ValueError('Unadmitted Mac SDK')
        common=['-swift-version','5','-strict-concurrency=complete','-warnings-as-errors','-O','-I',str(ROOT/'apple/include')]
        binary=output/'teacher';archive=native/'Toklet.xcframework/macos-arm64/libtoklet.a'
        execute(['xcrun','--sdk','macosx','swiftc','-sdk',sdk,'-module-cache-path',str(output/'MacModuleCache'),
                 '-target','arm64-apple-macos15.0',*common,*[str(p) for p in sources],str(archive),
                 '-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security',
                 '-lobjc','-liconv','-lresolv','-lc++','-o',str(binary)],'mac-compile')
        merge({str(binary.relative_to(ROOT)):package.sha(binary)})
        for backend,extra in [('cpu',[]),('metal',['metal'])]:
            stdout=execute([str(binary),str(ROOT/'artifacts/m7-native-tiny-v2'),*extra],backend)
            checked[backend]=validate_output(package.small(stdout),backend,oracle)
        for sdkname,target in [('iphoneos','arm64-apple-ios18.0'),('iphonesimulator','arm64-apple-ios18.0-simulator')]:
            sdk=package.small(execute(['xcrun','--sdk',sdkname,'--show-sdk-path'],sdkname+'-sdk')).strip()
            module=output/(sdkname+'.swiftmodule')
            execute(['xcrun','--sdk',sdkname,'swiftc','-sdk',sdk,'-module-cache-path',str(output/(sdkname+'-ModuleCache')),
                     '-target',target,*common,'-parse-as-library','-emit-module','-module-name','TokletTeacherDiagnostics',
                     '-emit-module-path',str(module),str(sources[0]),str(sources[1])],sdkname+'-compile')
            for p in output.glob(sdkname+'.*'):merge({str(p.relative_to(ROOT)):package.sha(p)})
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        package.signal.signal(package.signal.SIGTERM,previous)
        try:guard()
        except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
    passed=primary is None and not endpoints and len(commands)==10 and len(checked)==2
    record=dict(format='toklet-m7-native-teacher-check-v1',passed=passed,collection_root=str(ROOT),bindings=pins,commands=commands,
                checked=checked,independent_tiny_losses=oracle,controlled_environment=env,primary_error=primary,
                endpoint_errors=endpoints,full_quality_qualified=False,physical_device_qualified=False,m7_complete=False)
    with (output/'checks.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])))
