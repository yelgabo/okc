"""Isolated native-runtime validation; never writes live T3/Codex state.

Uses the existing managed provider's auth command in memory, without logging it.
Published output contains only status, allowlisted markers, and error codes.
"""
import ctypes
import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading

root = Path(tempfile.mkdtemp(prefix='t3-history-recovery-'))
native_id = '01a19974-57c5-7781-ab5c-2f71b614ec36'
source = next((Path(tempfile.gettempdir()) / 't3-provider-repro-20261101/private').glob(f'*{native_id}.jsonl'))
sessions = root / 'sessions/2026/11/01'
sessions.mkdir(parents=True)
shutil.copy2(source, sessions / source.name)
cache = Path.home() / '.codex/models_cache.json'
if cache.exists(): shutil.copy2(cache, root / cache.name)
workspace = root / 'workspace'
workspace.mkdir()
cmdline = subprocess.check_output(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Process -Filter "ProcessId = 49376").CommandLine'], text=True).strip()
parse = ctypes.windll.shell32.CommandLineToArgvW
parse.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
parse.restype = ctypes.POINTER(ctypes.c_wchar_p)
count = ctypes.c_int()
ptr = parse(cmdline, ctypes.byref(count))
args = [ptr[i] for i in range(count.value)]
ctypes.windll.kernel32.LocalFree(ptr)
launch = [args[0], 'app-server']
for i, arg in enumerate(args[:-1]):
    if arg == '-c' and args[i+1].split('=', 1)[0] in (
        'model_provider', 'model_providers.openai_token_sharing',
        'features.api_key_model_discovery', 'features.multi_agent', 'features.multi_agent_v2',
    ):
        launch.extend(['-c', args[i+1]])
assert any('model_providers.openai_token_sharing=' in x for x in launch)
env = {**os.environ, 'CODEX_HOME': str(root)}
for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'ACCESS_TOKEN', 'T3CODE_CODEX_LAUNCH_ARGS'):
    env.pop(key, None)
p = subprocess.Popen(launch, cwd=workspace, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW)
messages = queue.Queue()
def read():
    for line in p.stdout:
        try: messages.put(json.loads(line))
        except ValueError: pass
threading.Thread(target=read, daemon=True).start()
seq = 0
def send(method, params, request=True):
    global seq
    seq += 1
    body = dict(method=method, params=params)
    if request: body['id'] = seq
    p.stdin.write(json.dumps(body) + '\n'); p.stdin.flush()
    return seq
def rpc(method, params):
    ident = send(method, params)
    while True:
        msg = messages.get(timeout=90)
        if msg.get('id') == ident:
            if 'error' in msg: raise RuntimeError(f'{method} RPC rejected (code {msg["error"].get("code")})')
            return msg['result']
def turn(thread_id, text):
    send('turn/start', dict(threadId=thread_id, input=[dict(type='text', text=text, text_elements=[])], model='gpt-6-astra', effort='high', approvalPolicy='never', sandboxPolicy=dict(type='readOnly')))
    markers=[]
    while True:
        msg=messages.get(timeout=180)
        method=msg.get('method')
        data=msg.get('params', {})
        if method == 'item/completed':
            item=data.get('item', {})
            if item.get('type') == 'agentMessage':
                value=item.get('text','').strip()
                markers.append(value if value in ('BASELINE_OK','RECOVERY_OK','FOLLOWUP_OK') else '[non-marker assistant output]')
        if method == 'turn/completed':
            result=data['turn']
            raw=json.dumps(result.get('error'))
            code='subscription_sharing_unsupported_capability' if 'subscription_sharing_unsupported_capability' in raw else None
            return dict(status=result['status'], error_code=code, assistant_markers=markers)
result=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), codex='0.156.1', model='gpt-6-astra', provider='openai_token_sharing', isolated_home=str(root), source_native_thread=native_id)
try:
    rpc('initialize', dict(clientInfo=dict(name='t3_recovery_validation', version='1.0'), capabilities=dict(experimentalApi=True)))
    send('initialized', {}, request=False)
    params=dict(cwd=str(workspace), model='gpt-6-astra', approvalPolicy='never', sandbox='read-only', developerInstructions='This is an isolated recovery check. Do not call any tools or spawn agents. Answer only the requested marker.')
    resumed=rpc('thread/resume', dict(threadId=native_id, **params))
    result['before']=turn(resumed['thread']['id'], 'huh')
    print(json.dumps(dict(before=result['before'])), flush=True)
    fresh=rpc('thread/start', params)
    result['replacement_native_thread']=fresh['thread']['id']
    result['after']=turn(fresh['thread']['id'], 'Conversation history carried forward as text:\nUser: Reply exactly BASELINE_OK without tools.\nAssistant: BASELINE_OK\nUser: Run one worker.\nProvider: unsupported capability; no assistant answer.\nUser: huh\nProvider: unsupported capability; no assistant answer.\n\nCurrent user request: What exact token did you reply before the failed worker attempt? Reply only that token without tools.')
    print(json.dumps(dict(after=result['after'])), flush=True)
    result['followup']=turn(fresh['thread']['id'], 'Reply exactly FOLLOWUP_OK without tools.')
    print(json.dumps(dict(followup=result['followup'])), flush=True)
finally:
    p.terminate()
    p.wait(timeout=15)
    (Path(tempfile.gettempdir()) / 't3-recovery-runtime.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
