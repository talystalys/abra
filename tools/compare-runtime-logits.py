#!/usr/bin/env python3
"""Compare every finite logit and exact greedy token IDs; fail on mismatch.

Frozen tolerance: abs <= 0.001 + 0.0001 * abs(reference) for every logit.
Record bit equality separately. Uses stdlib; binary dumps are temporary.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct

def read(path):
    data = path.read_bytes()
    magic, vocab, steps = struct.unpack('<III', data[:12])
    if magic != 0x3149414c or len(data) != 12 + 4 * vocab * steps:
        raise SystemExit('Invalid dump shape/header')
    values = array('f')
    values.frombytes(data[12:])
    return (vocab, steps), values, hashlib.sha256(data).hexdigest()

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reference', type=Path)
p.add_argument('candidate', type=Path)
p.add_argument('reference_tokens', type=Path)
p.add_argument('candidate_tokens', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
shape, ref, ref_sha = read(a.reference)
other_shape, test, test_sha = read(a.candidate)
if shape != other_shape:
    raise SystemExit('Mismatched vocabulary/step dimensions')
rt = json.loads(a.reference_tokens.read_text())
ct = json.loads(a.candidate_tokens.read_text())
errors, violations, squared = 0.0, 0, 0.0
for x, y in zip(ref, test):
    if not math.isfinite(x) or not math.isfinite(y):
        raise SystemExit('Nonfinite logit')
    delta = abs(x - y)
    errors = max(errors, delta)
    squared += delta * delta
    violations += delta > 0.001 + 0.0001 * abs(x)
tokens_match = rt['tokens'] == ct['tokens']
prompts_match = rt['prompt_tokens'] == ct['prompt_tokens']
report = dict(shape=shape, logits_checked=len(ref), max_absolute_error=errors,
              rms_error=math.sqrt(squared / len(ref)), tolerance_violations=violations,
              absolute_tolerance=0.001, relative_tolerance=0.0001,
              tokens_identical=tokens_match, prompt_lengths_identical=prompts_match,
              bit_identical=ref_sha == test_sha, reference_sha256=ref_sha,
              candidate_sha256=test_sha, accepted=violations == 0 and tokens_match and prompts_match)
a.out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if report['accepted'] else 1)
