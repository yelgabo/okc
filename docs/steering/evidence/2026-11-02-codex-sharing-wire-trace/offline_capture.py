"""Capture copied-root request serialization against localhost, never a real provider."""
import datetime
import hashlib
import http.server
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading

HOME = pathlib.Path.home()
OUTPUT = pathlib.Path(__file__).resolve().parent
CLI = HOME / 'AppData/Roaming/npm/node_modules/@openai/codex/node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe'
SOURCE = HOME / '.codex/sessions/2026/11/02/rollout-2026-11-02T01-54-44-01a19bae-c402-79c3-a27f-baff9e5fd2ac.jsonl'
TEMP = pathlib.Path(tempfile.mkdtemp(prefix='codex-sharing-wire-'))
DEST = TEMP / 'sessions/2026/11/02' / SOURCE.name
DEST.parent.mkdir(parents=True)
shutil.copyfile(SOURCE, DEST)
CATALOG = json.loads((HOME / '.codex/models_cache.json').read_text(encoding='utf-8'))
RESULT = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'source_rollout': SOURCE.name, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'cli_version': subprocess.check_output([str(CLI), '--version'], text=True).strip(),
          'requests': [], 'raw_request_saved': False, 'live_state_modified': False}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        RESULT['requests'].append({'method': 'GET', 'path': self.path})
        data = json.dumps({'models': CATALOG['models']}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        data = self.rfile.read(int(self.headers['Content-Length']))
        body = json.loads(data)
        items = body.get('input', [])
        agents = []
        for item in items:
            if item.get('type') != 'agent_message':
                continue
            agents.append({'type': item['type'], 'keys': list(item),
                           'author': item.get('author'), 'recipient': item.get('recipient'),
                           'content': [{'type': part['type'],
                                        'characters': len(part.get('text', part.get('encrypted_content', '')))}
                                       for part in item.get('content', [])]})
        RESULT['requests'].append({'method': 'POST', 'path': self.path,
                                   'body_bytes': len(data), 'body_sha256': hashlib.sha256(data).hexdigest(),
                                   'keys': list(body), 'model': body.get('model'),
                                   'store': body.get('store'), 'stream': body.get('stream'),
                                   'service_tier': body.get('service_tier'),
                                   'input_types': [item.get('type', 'message') for item in items],
                                   'agent_messages': agents})
        self.send_response(400)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': {'message': 'Local serialization diagnostic stops before inference.',
                                              'type': 'invalid_request_error', 'param': 'input',
                                              'code': 'local_diagnostic_stop'}}).encode())


server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
worker = threading.Thread(target=server.serve_forever, daemon=True)
worker.start()
port = server.server_port
provider = ('model_providers.openai_token_sharing={name="Local diagnostic",'
            f'base_url="http://127.0.0.1:{port}/v1",'
            f'model_catalog_url="http://127.0.0.1:{port}/v1/models",'
            'env_key="LOCAL_DIAGNOSTIC_TOKEN",wire_api="responses",requires_openai_auth=false,'
            'supports_websockets=false,request_max_retries=0,stream_max_retries=0}')
args = [str(CLI), 'exec', '-C', str(TEMP), '--skip-git-repo-check', '--ignore-user-config',
        '--ignore-rules', '--json', '-c', 'model_provider="openai_token_sharing"', '-c', provider,
        '-c', 'features.multi_agent=false', '-c', 'features.multi_agent_v2=false',
        '-c', 'agents.enabled=false', '-c', 'features.api_key_model_discovery=true',
        '-c', 'analytics.enabled=false', 'resume', '--all',
        '01a19bae-c402-79c3-a27f-baff9e5fd2ac', 'LOCAL_SCHEMA_CAPTURE_ONLY']
env = {key: os.environ[key] for key in ['SystemRoot', 'WINDIR', 'COMSPEC', 'PATH', 'TEMP', 'TMP',
                                       'USERPROFILE', 'APPDATA', 'LOCALAPPDATA'] if key in os.environ}
env.update(CODEX_HOME=str(TEMP), LOCAL_DIAGNOSTIC_TOKEN='dummy-local-only', CODEX_ROLLOUT_TRACE_ROOT='')
try:
    completed = subprocess.run(args, cwd=TEMP, env=env, capture_output=True, text=True, timeout=20)
    RESULT['exit_code'] = completed.returncode
    RESULT['stdout_events'] = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{')]
    RESULT['stderr'] = completed.stderr
finally:
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)
RESULT['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
RESULT['temp_home_retained'] = str(TEMP)
RESULT['safety'] = ['Only copied ROOT history resumed; no agent-spawn request.',
                    'Localhost mock returns HTTP 400 before any inference or tool output.',
                    'Dummy token only; no auth files or live configuration copied.',
                    'agents.enabled=false added for diagnostic safety, not a claimed history repair.']
posts = [request for request in RESULT['requests'] if request['method'] == 'POST']
assert len(posts) == 1 and posts[0]['path'] == '/v1/responses'
assert posts[0]['agent_messages'], 'No native agent item serialized'
assert all(event['type'] not in ['item.started', 'item.completed'] for event in RESULT['stdout_events'])
(OUTPUT / 'offline-capture.json').write_text(json.dumps(RESULT, indent=2) + chr(10), encoding='utf-8')
print(json.dumps({'post_count': len(posts), 'input_count': len(posts[0]['input_types']),
                  'agent_messages': posts[0]['agent_messages'], 'exit_code': RESULT['exit_code']}))
