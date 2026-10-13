"""The automatic long-run handoff must reject incomplete or unhealthy gates."""
import copy

import pytest

from train.resume_three_gpu import validate_gate


def evidence():
    before = dict(step=4000, world_size=3, total_tokens_seen=1048838144,
                  run_signature={'total_steps': 38147, 'world_size': 3})
    after = copy.deepcopy(before)
    after.update(step=4200, total_tokens_seen=before['total_tokens_seen'] + 200 * 264192,
                 val_lm_loss=3.0)
    lines = ['Resumed successfully at Step 4001 (Tokens seen: 1,048,838,144)']
    for step in [4001, 4002, *range(4010, 4201, 10)]:
        lines.append(f'[Step {step:05d}/38147] loss 4.0 | lm 3.0 | mtp 3.3 | '
                     'step 102.0s | tok/s 2,590 | MFU 0.7% | '
                     'mem max alloc 19.5 GiB / reserved 20.6 GiB | dead 0.0% | imb 1.4 | scale 1024 | skips 0')
    lines.append('Bounded continuation complete.')
    return '\n'.join(lines), before, after


def test_complete_continuation_gate_passes():
    report = validate_gate(*evidence())
    assert report['status'] == 'passed'
    assert report['final_step'] == 4200


@pytest.mark.parametrize('change', [
    lambda s: s.replace('Bounded continuation complete.', ''),
    lambda s: s.replace('Step 4001 (', 'Step 0 ('),
    lambda s: s + '\n[!] Step 4050 SKIPPED: non-finite gradient',
    lambda s: s.replace('reserved 20.6', 'reserved 31.0'),
    lambda s: s.replace('lm 3.0', 'lm nan'),
    lambda s: s.replace('[Step 04200/', '[Step 04199/'),
])
def test_bad_or_incomplete_gate_stops_handoff(change):
    log, before, after = evidence()
    with pytest.raises(ValueError):
        validate_gate(change(log), before, after)


def test_signature_or_progress_change_stops_handoff():
    for key, value in [('total_tokens_seen', 1), ('run_signature', {})]:
        log, before, after = evidence()
        after[key] = value
        with pytest.raises(ValueError):
            validate_gate(log, before, after)
