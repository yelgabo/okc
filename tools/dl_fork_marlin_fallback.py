"""Use the baseline library path when the repaired native experts cannot run."""
import ast


def patch_fallback(source):
    old = """    if st['_fork_fallback_kind']=='owned': return _fork_owned_moe_for(M,st)
    return st['_fork_marlin_adapter']"""
    if source.count(old) != 1:
        raise ValueError('unexpected expert fallback selector')
    source = source.replace(old, "    return st['_fork_marlin_adapter']")

    start = source.index('def _fork_install_fallback(')
    end = source.index('def _ma_prepare(', start)
    block = source[start:end]
    before = block.index('    installed = _fork_install_owned_moe(')
    warm = block.index('        _fork_prepare_marlin(')
    after = block.index('\n    torch.cuda.synchronize()', warm)
    body = block[warm:after]
    assert all(line.startswith('        ') for line in body.splitlines())
    body = '\n'.join(line[4:] for line in body.splitlines())
    block = block[:before] + "    st['_fork_fallback_kind'] = 'Marlin'\n" + body + block[after:]
    source = source[:start] + block + source[end:]

    # The optional owned backend occupied a contiguous, independently appended
    # module. Require that no surviving host code still references its namespace.
    start = source.index('_OWNED_MOE_DEVICE=')
    end = source.index('_g1a_prepare=prepare', start)
    source = source[:start] + source[end:]
    tree = ast.parse(source)
    orphan = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
              and n.id.startswith(('_OWNED_', '_owned_', '_fork_owned_', '_fork_install_owned_', '_OwnedLayerMoe'))}
    if orphan:
        raise ValueError(f'owned fallback still referenced: {orphan}')
    return source
