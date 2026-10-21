"""Strict Mac compilation and actual tiny CPU/Metal measurement/failure checks.

Uses an already reviewed native package; does not sign or launch an iOS app.
The fixture and every compiled input are admitted before compilation and checked
again after the bounded owned child returns. Outputs are always exclusive.
"""
import json
from pathlib import Path
import sys
import m7_apple_package as package
import m7_apple_integration as integration

ROOT = package.ROOT


def validate_output(text):
    def records(prefix):
        return [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]
    results = records('MEASUREMENT_CHECK_RESULT_JSON=')
    samples = records('MEASUREMENT_CHECK_SAMPLE_JSON=')
    summaries = records('MEASUREMENT_CHECKS_JSON=')
    if len(results) != 2 or len(samples) != 26 or len(summaries) != 1:
        raise ValueError('Incomplete actual measurement checks')
    for backend, result, group in zip(('cpu','metal'), results, (samples[:13],samples[13:])):
        if (result['backend'] != backend or result['passed'] is not True
            or result.get('error') is not None or result['cleanupErrors'] or result['endpointErrors']
            or result['closed'] is not True or result['released'] is not True
            or result['warmupsCompleted'] != 3 or result['measuredCompleted'] != 10
            or result['physicalDeviceQualified'] is not False or result['m7Complete'] is not False
            or result['filesBefore'] != result['filesAfter']):
            raise ValueError('Backend result/cleanup differs')
        total = 0
        for index, sample in enumerate(group):
            if (sample['index'] != index or sample['role'] != ('warmup' if index < 3 else 'measured')
                or sample['tokens'] != [47,47] or len(sample['events']) != 2
                or (sample['history'],sample['evaluated'],sample['pending'],sample['reason'],sample['status']) != (6,5,1,1,0)
                or sample['footprint']['physicalBytes'] <= 0
                or sample['footprint']['lifetimePeakPhysicalBytes'] < sample['footprint']['physicalBytes']):
                raise ValueError('Sample oracle/state/memory differs')
            previous = 0
            for offset,event in enumerate(sample['events']):
                if (event['index'] != offset or not 0 <= event['nativeElapsedSeconds'] < 1200
                    or event['nativeElapsedSeconds'] < previous
                    or not sample['submittedUptimeNanoseconds'] <= event['callbackUptimeNanoseconds'] <= sample['completedUptimeNanoseconds']):
                    raise ValueError('Sample clock differs')
                previous = event['nativeElapsedSeconds']
            if index >= 3: total += sample['events'][-1]['nativeElapsedSeconds']
        if total != result['sumNativeLastEventSeconds']:
            raise ValueError('Measured duration sum differs')
    summary = summaries[0]
    if summary['passed'] is not True or len(summary['cases']) != 15 or summary['physicalDeviceQualified'] is not False:
        raise ValueError('Failure-case checks incomplete')
    return dict(results=results,sample_count=len(samples),summary=summary)


def verify_inputs(pins):
    for name,pin in pins.items():
        if package.sha(ROOT/name) != pin: raise ValueError('Changed admitted input: '+name)


def main(native, expected, output):
    native=native.resolve();output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'): raise ValueError('Expected repository artifact destination')
    output.mkdir();logs=output/'logs';logs.mkdir();commands=[];pins={};endpoints=[];primary=None;checked=None
    env,_=package.build_environment();env['TMPDIR']=str(output)
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    def execute(args,label):
        verify_inputs(pins)
        row=package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,180)
        commands.append(row)
        if row['primary_error'] or row['cleanup_errors']: raise RuntimeError(row['primary_error'] or repr(row['cleanup_errors']))
        return ROOT/row['stdout']
    try:
        integration.merge(pins,integration.admit_build(native,expected))
        helpers=[Path(package.__file__).resolve(),Path(integration.__file__).resolve(),
                 Path(__file__).resolve(),Path(__file__).with_name('test_m7_native_measurement_check.py')]
        integration.merge(pins,{str(p.relative_to(ROOT)):package.sha(p) for p in helpers})
        verify_inputs(pins)
        archive=native/'Toklet.xcframework/macos-arm64/libtoklet.a'
        sources=[ROOT/'apple/Harness/NativeWorker.swift',ROOT/'apple/Harness/NativeMeasurement.swift',ROOT/'apple/tests/swift/MeasurementMain.swift']
        fixture=ROOT/'artifacts/m7-native-tiny-v2'
        paths=[*sources,Path(__file__).resolve(),archive,fixture/'config.json',fixture/'model.safetensors']
        integration.merge(pins,{str(p.relative_to(ROOT)):package.sha(p) for p in paths})
        for name,pin in integration.FIXTURE.items():
            if package.sha(fixture/Path(name).name)!=pin:raise ValueError('Independent tiny fixture differs')
        binary=output/'check'
        execute(['xcodebuild','-version'],'xcode-version')
        sdk=package.small(execute(['xcrun','--sdk','macosx','--show-sdk-path'],'sdk-path')).strip()
        if sdk != '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk':
            raise ValueError('Unadmitted SDK path')
        execute(['xcrun','--sdk','macosx','swiftc','--version'],'swift-version')
        command=['xcrun','--sdk','macosx','swiftc','-sdk',sdk,
            '-module-cache-path',str(output/'ModuleCache'),'-swift-version','5','-strict-concurrency=complete','-warnings-as-errors',
            '-target','arm64-apple-macos15.0','-O','-I',str(ROOT/'apple/include'),*[str(p) for p in sources],str(archive),
            '-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-framework','Security',
            '-lobjc','-liconv','-lresolv','-lc++','-o',str(binary)]
        for label,args in [('compile',command),('checks',[str(binary),str(fixture),str(output/'cases')])]:
            execute(args,label)
            if label=='compile':pins[str(binary.relative_to(ROOT))]=package.sha(binary)
        checked=validate_output(package.small(logs/'checks.stdout'))
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        package.signal.signal(package.signal.SIGTERM,previous)
        for name,pin in pins.items():
            try:
                if package.sha(ROOT/name)!=pin:endpoints.append(name)
            except OSError:endpoints.append('Unavailable: '+name)
    passed=primary is None and not endpoints and checked is not None
    record=dict(format='toklet-m7-native-measurement-check-v1',passed=passed,bindings=pins,commands=commands,controlled_environment=env,
                checked=checked,primary_error=primary,endpoint_errors=endpoints,physical_device_qualified=False,m7_complete=False)
    (output/'checks.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])))
