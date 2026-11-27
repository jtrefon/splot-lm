"""REP v0.1 structural evidence gate; not a proof of measurement truth."""
import argparse
import hashlib
import json
from pathlib import Path
import re


CHECK = {'passed', 'failed', 'blocked', 'not_run'}
REQUIRED = {'summary.md', 'environment.json', 'metrics.jsonl', 'execution.log'}
HEX40 = re.compile(r'[0-9a-f]{40}')
HEX64 = re.compile(r'[0-9a-f]{64}')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject, object_pairs_hook=unique)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory, task, source, expected_checks):
    d = Path(directory)
    require(d.is_dir() and not d.is_symlink(), 'attempt directory required')
    require((d/'result.json').is_file() and not (d/'result.json').is_symlink(), 'result.json required')
    r = read_json(d/'result.json')
    require(isinstance(r, dict) and type(r.get('schema_version')) is int and r['schema_version'] == 1, 'schema version must be 1')
    require(re.fullmatch(r'T[0-9]{4}', task) and r.get('task_id') == task, 'task mismatch')
    require(type(r.get('attempt')) is int and r['attempt'] >= 1, 'invalid attempt')
    require(d.name == f"attempt-{r['attempt']:03d}" and d.parent.name == task, 'attempt path mismatch')
    for k in ('task_commit', 'source_commit'):
        require(isinstance(r.get(k), str) and HEX40.fullmatch(r[k]), 'invalid ' + k)
    require(r['source_commit'] == source, 'source pin mismatch')
    require('tested_commit' in r and r['tested_commit'] in (None, source), 'tested commit mismatch')
    require(isinstance(r.get('deviations'), list) and all(isinstance(v, str) and v.strip() for v in r['deviations']), 'invalid deviations')
    require(isinstance(r.get('limitations'), list) and all(isinstance(v, str) and v.strip() for v in r['limitations']), 'invalid limitations')
    require(isinstance(r.get('external_artifacts'), list), 'external_artifacts list required')
    for a in r['external_artifacts']:
        require(isinstance(a, dict) and isinstance(a.get('name'), str) and a['name'].strip() and isinstance(a.get('sha256'), str) and HEX64.fullmatch(a['sha256']) and type(a.get('bytes')) is int and a['bytes'] >= 0, 'invalid external artifact')
    files = r.get('files')
    require(isinstance(files, dict) and REQUIRED <= files.keys(), 'missing required evidence entries')
    actual = set()
    for p in d.iterdir():
        require(p.is_file() and not p.is_symlink(), 'only regular flat evidence files allowed')
        if p.name != 'result.json': actual.add(p.name)
    require(actual == set(files), 'unlisted or missing evidence files')
    for name, digest in files.items():
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name) and name != 'result.json', 'unsafe evidence filename')
        require(isinstance(digest, str) and HEX64.fullmatch(digest) and sha(d/name) == digest, 'evidence hash mismatch: ' + name)
    require((d/'summary.md').stat().st_size > 0 and (d/'execution.log').stat().st_size > 0, 'empty summary or log')
    require(isinstance(read_json(d/'environment.json'), dict), 'environment must be object')
    for line in (d/'metrics.jsonl').read_text(encoding='utf-8').splitlines():
        require(isinstance(json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v))), dict), 'metrics line must be JSON object')
    checks = r.get('checks')
    require(isinstance(checks, dict) and set(checks) == set(expected_checks), 'missing or unexpected checks')
    statuses = []
    for key, c in checks.items():
        require(isinstance(c, dict) and c.get('status') in CHECK, 'invalid check: ' + key)
        require(isinstance(c.get('detail'), str) and c['detail'].strip(), 'missing check detail: ' + key)
        ev = c.get('evidence')
        require(isinstance(ev, list) and all(isinstance(v, str) and v in files for v in ev), 'invalid evidence reference: ' + key)
        if c['status'] in ('passed', 'failed', 'blocked'): require(bool(ev), 'missing check evidence: ' + key)
        statuses.append(c['status'])
    status = 'failed' if 'failed' in statuses else 'passed' if all(s == 'passed' for s in statuses) else 'blocked'
    require(r.get('status') == status, 'overall status inconsistent with checks')
    if status == 'passed': require(r['tested_commit'] == source and not r['deviations'], 'pass requires exact revision and no deviations')
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory'); p.add_argument('--task', required=True); p.add_argument('--source', required=True)
    p.add_argument('--checks', required=True, help='comma-separated exact check names from task')
    a = p.parse_args()
    try:
        status = validate(a.directory, a.task, a.source, a.checks.split(','))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        p.exit(1, f'INVALID: {exc}\n')
    print('STRUCTURALLY VALID:', status, '(not scientific acceptance)')


if __name__ == '__main__': main()
