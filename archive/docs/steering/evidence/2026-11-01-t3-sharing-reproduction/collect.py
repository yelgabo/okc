"""Read-only evidence extraction; raw snapshots stay in the local temp directory.

Run from this folder or by absolute path. No auth files, databases, or credentials
are read. Published JSON uses explicit field allowlists; message bodies omitted.
"""
import collections
import datetime
import hashlib
import json
import os
import pathlib
import re
import struct
import subprocess
import tempfile

OUT = pathlib.Path(__file__).resolve().parent
HOME = pathlib.Path.home()
PRIVATE = pathlib.Path(tempfile.gettempdir()) / 't3-provider-repro-20261101' / 'private'
PRIVATE.mkdir(parents=True, exist_ok=True)
LABELS = {
    '01a19968-9e4f-74d1-b55e-25d1a3bfe702': 'original_parent',
    '01a1996a-6992-7c91-b571-35da58cfdcbb': 'original_worker',
    '01a19974-57c5-7781-ab5c-2f71b614ec36': 'reproduction_parent',
    '01a19974-f297-7791-8750-95c36b7ba898': 'reproduction_worker',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def snapshot(path):
    data = path.read_bytes()
    (PRIVATE / path.name).write_bytes(data)
    return data, {'name': path.name, 'bytes': len(data), 'sha256': digest(data)}


def pick(data, keys):
    return {key: data[key] for key in keys if key in data}


manifest = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'raw_snapshots': []}
log = HOME / '.t3/userdata/logs/provider/events.40356bc9-4e84-4f44-81ab-572185175234.log'
raw, meta = snapshot(log)
manifest['raw_snapshots'].append(meta)
events = []
parsed = []
for line_number, line in enumerate(raw.decode('utf-8').split('\n'), 1):
    try:
        record = json.loads(line[line.index('{'):])
    except (ValueError, TypeError):
        continue
    event = record.get('event', {})
    payload = event.get('payload', {})
    parsed.append((line_number, line[1:25], event.get('direction'), payload))

start_requests = {}
for line_number, timestamp, direction, payload in parsed:
    method = payload.get('method')
    params = payload.get('params', {})
    result = payload.get('result', {})
    if method in ('thread/start', 'thread/resume'):
        start_requests[payload.get('id')] = (line_number, timestamp, method, params)
    if isinstance(result, dict) and isinstance(result.get('thread'), dict):
        tid = result['thread'].get('id')
        if tid in LABELS and payload.get('id') in start_requests:
            n, t, m, p = start_requests[payload['id']]
            events.append({'line': n, 'timestamp': t, 'direction': 'outgoing', 'method': m,
                           'thread': LABELS[tid], 'params': pick(p, ['threadId', 'model']),
                           'config_keys': list(p.get('config', {}))})
            events.append({'line': line_number, 'timestamp': timestamp, 'direction': direction,
                           'method': m + ':result', 'thread': LABELS[tid],
                           'result': pick(result, ['model', 'modelProvider', 'reasoningEffort'])})
    if method == 'initialize':
        events.append({'line': line_number, 'timestamp': timestamp, 'method': method,
                       'client': params.get('clientInfo')})
    if payload.get('id') == 1 and isinstance(result, dict) and 'userAgent' in result:
        events.append({'line': line_number, 'timestamp': timestamp, 'method': 'initialize:result',
                       'userAgent': result['userAgent']})
    tid = params.get('threadId')
    if tid not in LABELS:
        continue
    base = {'line': line_number, 'timestamp': timestamp, 'direction': direction,
            'method': method, 'thread': LABELS[tid], 'threadId': tid}
    if method == 'error':
        # Raw error text has no prompt content or credential material.
        events.append({**base, 'params': params})
    elif method == 'turn/start':
        safe = pick(params, ['threadId', 'model', 'effort'])
        safe['input_types'] = [x.get('type') for x in params.get('input', [])]
        safe['input_count'] = len(params.get('input', []))
        safe['plain_huh'] = any(x.get('text') == 'huh' for x in params.get('input', []))
        events.append({**base, 'params': safe})
    elif method == 'turn/completed':
        events.append({**base, 'turn': pick(params.get('turn', {}),
                      ['id', 'status', 'error', 'startedAt', 'completedAt', 'durationMs'])})
    elif method in ('item/started', 'item/completed') and params.get('item', {}).get('type') == 'subAgentActivity':
        item = params['item']
        events.append({**base, 'turnId': params.get('turnId'),
                      'item': pick(item, ['type', 'id', 'status', 'tool', 'model', 'reasoningEffort',
                                         'senderThreadId', 'receiverThreadIds'])})
save('protocol-extract.json', sorted(events, key=lambda x: x['line']))

histories = []
for tid, label in LABELS.items():
    matches = list((HOME / '.codex/sessions/2026/11/01').glob('*' + tid + '.jsonl'))
    assert len(matches) == 1, (tid, len(matches))
    data, meta = snapshot(matches[0])
    manifest['raw_snapshots'].append(meta)
    rows = []
    for n, line in enumerate(data.decode('utf-8').split('\n'), 1):
        if not line:
            continue
        entry = json.loads(line)
        p = entry.get('payload', {})
        if entry.get('type') != 'response_item':
            continue
        row = {'line': n, 'timestamp': entry.get('timestamp'),
               **pick(p, ['type', 'id', 'role', 'namespace', 'name', 'call_id', 'author', 'recipient'])}
        if 'content' in p:
            content = p['content']
            row['content_types'] = [c.get('type') for c in content if isinstance(c, dict)] if isinstance(content, list) else []
            row['content_omitted'] = True
        if p.get('type') == 'function_call':
            try:
                row['arguments'] = pick(json.loads(p['arguments']), ['task_name', 'timeout_ms', 'fork_turns'])
            except (ValueError, KeyError):
                pass
        rows.append(row)
    histories.append({'label': label, 'thread_id': tid, 'source': meta, 'response_items': rows})
save('rollout-structure.json', histories)

resources = HOME / 'AppData/Local/Programs/t3code/resources'
app_info = {}
for archive_name in ['app.asar', 'server.asar']:
    archive = resources / archive_name
    data = archive.read_bytes()
    manifest[archive_name] = {'bytes': len(data), 'sha256': digest(data)}
    _, header_size, _, json_size = struct.unpack('<IIII', data[:16])
    tree = json.loads(data[16:16 + json_size])
    entry = tree['files']['package.json']
    begin = 8 + header_size + int(entry['offset'])
    app_info[archive_name] = json.loads(data[begin:begin + entry['size']])
    if archive_name == 'server.asar':
        files = tree['files']['apps']['files']['server']['files']['dist']['files']
        name = next(k for k in files if k.startswith('binCli-') and k.endswith('.mjs'))
        entry = files[name]
        begin = 8 + header_size + int(entry['offset'])
        js = data[begin:begin + entry['size']]
        lines = js.decode('utf-8').split('\n')
        manifest['server_bundle'] = {'name': name, 'bytes': len(js), 'sha256': digest(js)}
        snippets = []
        for start_marker, end_marker in [
            ('// src/provider/CodexManagedErrors.ts', '// ../../packages/effect-codex-app-server'),
            ('const managedCodexLaunchArgs =', 'const makeCodexManagedRuntime ='),
            ('function withCodexManagedFailure(', 'function codexRetryTurnItem('),
        ]:
            start = next(i for i, s in enumerate(lines) if s.startswith(start_marker))
            end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith(end_marker))
            snippets.append({'start_line': start + 1, 'end_line': end, 'source': '\n'.join(lines[start:end])})
        save('installed-source-excerpts.json', {'bundle': manifest['server_bundle'], 'excerpts': snippets})

cli = HOME / '.t3/tools/codex/0.156.1/bin/codex.exe'
version = subprocess.check_output([str(cli), '--version'], text=True).strip()
feature_result = subprocess.check_output([str(cli), '-c', 'features.multi_agent=false', '-c',
                                         'features.multi_agent_v2=false', 'features', 'list'], text=True)
process_script = r'''Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -eq 'C:\Users\gretc\.t3\tools\codex\0.156.1\bin\codex.exe' } | ForEach-Object { [pscustomobject]@{ ProcessId=$_.ProcessId; HasAgentsEnabledOverride=$_.CommandLine.Contains('agents.enabled'); FeatureFlags=@([regex]::Matches($_.CommandLine,'features\.[A-Za-z0-9_]+\s*=\s*(true|false)') | ForEach-Object { $_.Value }) } } | ConvertTo-Json -Depth 3'''
process = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command', process_script], text=True))
catalog = json.loads((HOME / '.codex/models_cache.json').read_text(encoding='utf-8'))
selected = [pick(m, ['slug', 'multi_agent_version', 'multi_agent_reasoning_effort'])
            for m in catalog.get('models', []) if m.get('slug') in ['gpt-6-astra', 'gpt-6.1-sol']]
save('runtime.json', {
    'captured_utc': manifest['captured_utc'],
    'app': pick(app_info['app.asar'], ['name', 'version', 'buildVersion', 't3codeCommitHash']),
    'codex_version': version, 'process': process,
    'feature_check_command': 'codex -c features.multi_agent=false -c features.multi_agent_v2=false features list',
    'feature_check_selected_output': [line for line in feature_result.splitlines() if 'agent' in line or 'collab' in line],
    'local_catalog_selected_fields': selected,
    'catalog_note': 'Read current shared home cache; not a captured per-request remote catalog response.',
})
save('snapshot-manifest.json', manifest)
print(json.dumps({'protocol_records': len(events), 'histories': len(histories),
                  'private_snapshot_directory': str(PRIVATE), 'output_directory': str(OUT)}))
