from pathlib import Path
import ast, base64, hashlib, json, lzma, re, zlib

out = Path('.build/scout-2026-11-03-1405')
root = out / 'autoclicker-source'
folder = root / 'final/muse_decode_layer'
tree = ast.parse((folder / 'build.py').read_text())
constants = {n.targets[0].id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
wrapper = (root / 'problems/muse_decode_layer/cand_r3_best7_s22p1_final.py').read_bytes()
archived_wrapper = (root / 'runs/muse_decode_layer/20261027T013731Z_submit_210d40e0c9cb4b71.py').read_bytes()
assert wrapper == archived_wrapper
assert hashlib.sha256(wrapper).hexdigest() == constants['SHIPPED_SHA256']
match = re.search(rb"b85decode\('([^']+)'\)", wrapper)
assert match
unpacked = lzma.decompress(base64.b85decode(match.group(1)))
assert hashlib.sha256(unpacked).hexdigest() == constants['SOURCE_SHA256']
pattern = re.compile(r'_embedded\("([^"]+)"(, zlib=True)?\)')
used = set()

def resolve(match):
    path = match.group(1)
    target = (folder / 'src' / path).resolve()
    assert str(target).startswith(str((folder / 'src').resolve()) + '/')
    used.add(path)
    text = target.read_text()
    if path.endswith('.py'):
        text = pattern.sub(resolve, text)
    if match.group(2):
        used.add(path + '.b64')
        literal = Path(str(target) + '.b64').read_text()
        assert zlib.decompress(base64.b64decode(literal)).decode() == text
        return repr(literal)
    return repr(text)

layer = (folder / 'layer.py').read_text()
assert layer.count(constants['LOADER']) == 1
joined = pattern.sub(resolve, layer.replace(constants['LOADER'], '', 1)).encode()
assert joined == unpacked
all_files = {str(p.relative_to(folder / 'src')) for p in (folder / 'src').rglob('*') if p.is_file()}
assert used == all_files
manifest = json.loads((out / 'autoclicker-source-manifest.json').read_text())
for row in manifest:
    data = Path(row['local']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == row['blob']
receipt = json.loads((root / 'runs/muse_decode_layer/20261027T013731Z_submit_210d40e0c9cb4b71.json').read_text())
result = receipt['result']
record = dict(repository='AutoclickerI/muse-spark-open-kernels', pin='85aed1fb1365877b615542dd44980646816d6b26', source_files_verified=len(manifest), embedded_files=len(used), wrapper_sha256=constants['SHIPPED_SHA256'], expanded_sha256=constants['SOURCE_SHA256'], expanded_bytes=len(unpacked), byte_exact_static_join=True, foreign_code_executed=False, uploaded_receipt=dict(job=receipt['id'],created_at=receipt['created_at'],gpu=result['gpu'],score_us=result['score_us'],speedup=result['speedup'],speedup_ci=result['speedup_ci'],passed=result['passed'],flags=result['flags'],checks=result['checks'],content_hash=receipt['content_hash']), limits='Static data reconstruction and public Git-blob verification only. Receipt path contains matching wrapper hash prefix, but payload content_hash uses a different representation whose server binding was not independently checked. No compilation, numerical validation, GPU execution or reproduction. Stage README contains additional historical wrapper hashes; do not substitute them for the captured final file.')
(out / 'static-source-check.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k:record[k] for k in ['source_files_verified','embedded_files','byte_exact_static_join','foreign_code_executed','expanded_bytes']},indent=2))
