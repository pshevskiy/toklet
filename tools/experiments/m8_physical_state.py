"""Strict MLX state512 admission; reuses the unchanged exact-state oracle.

This checks diagnostic state and scalar score preservation, not whole-distribution
parity, OS lifecycle delivery, quality, performance or general device support.
"""
import json

import m7_native_state_check as state
import m7_physical_measurement as measurement
import m8_mlx_jobs as jobs


def admit_config(config,component=False):
    if type(config) is not dict:
        raise ValueError('Exact physical MLX state configuration required')
    model={'M8Inputs/qwen/Model':'qwen','M8Inputs/qwen8/Model':'qwen8','M8Inputs/smollm/Model':'smollm'}.get(config.get('modelFolder'))
    if model is None or jobs.encoded(config)!=jobs.encoded(jobs.job(model,'state-component' if component else 'state-512')[1]):
        raise ValueError('Frozen real MLX state512 recipe required')
    return model


def validate_console(text,config,signed,component=False):
    admit_config(config,component)
    markers={'TOKLET_INSTALLATION_JSON=':'proof','TOKLET_STATE_JSON=':'result'};rows=[]
    for line in text.splitlines():
        for prefix,kind in markers.items():
            if line.startswith(prefix):rows.append((kind,json.loads(line[len(prefix):])));break
        else:
            if line.startswith('TOKLET_'):raise ValueError('Unexpected MLX state marker')
    if [kind for kind,_ in rows]!=['proof','proof','result']:
        raise ValueError('Exact installed MLX state stream required')
    proofs=[rows[0][1],rows[1][1]];result=rows[2][1]
    if ([proof['phase'] for proof in proofs]!=['before','after']
        or any(proof['simulator'] is not False or proof['files']!=signed for proof in proofs)):
        raise ValueError('Installed physical MLX state bundle differs')
    # Legacy kind "small" selects the 512/chunk128/capacity1152 prefill witness,
    # not a weight representation. Backend and the full recipe remain explicit.
    state.validate_result(result,'metal-mlx','component' if component else 'small',config)
    for row in result['rows']:
        env=row['environment']
        if (not measurement.uint(env['applicationState'],32) or env['applicationState']!=0
            or not measurement.uint(env['thermalState'],32) or env['thermalState']>3
            or type(env['lowPowerMode']) is not bool
            or not measurement.uint(env['batteryState'],32) or env['batteryState'] not in range(1,4)
            or not measurement.nonnegative(env['batteryLevel']) or env['batteryLevel']>1):
            raise ValueError('Physical foreground/power/thermal state required')
    return dict(result=result,installed_proofs=proofs)
