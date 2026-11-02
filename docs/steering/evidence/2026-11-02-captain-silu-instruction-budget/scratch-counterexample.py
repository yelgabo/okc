import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[4]
source = (root / 'muse_decode_layer/cand/g1b4_stage_release.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
node = next(item for item in ast.parse(source.decode()).body if isinstance(item, ast.FunctionDef) and item.name == '_ma_make_scratch')
namespace = dict(torch=SimpleNamespace(zeros=lambda *args, **kwargs: object(), int32='int32', float32='float32', bfloat16='bf16'), _MA_EMAX=1, WGS_FIXF=1, WGS_XROWS=1, _df_inter=lambda value: value)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-public607-scratch-factory>', 'exec'), namespace)
production = namespace['_ma_make_scratch']('stub')
production.append(object())
private = namespace['_ma_make_scratch']('stub')
assert len(production) == 4 and len(private) == 3
assert 'scratch_factory(device)' in source.decode()
print('PASS actual isolated factory: production-only append gives 4; independently prepared private startup remains 3. CPU ABI counterexample only, not GPU execution.')
