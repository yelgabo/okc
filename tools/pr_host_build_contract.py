"""Validate explicit PR host flags against a retained target build diagnostic."""

import ast
import re


def standard(flags):
    selected = [flag for flag in flags if flag.startswith('-std=')]
    if not selected:
        raise ValueError('missing explicit language standard')
    match = re.fullmatch(r'-std=(?:c|gnu)\+\+(17|20|23|26)', selected[-1])
    if match is None:
        raise ValueError('unsupported language standard: ' + selected[-1])
    return int(match.group(1))


def contract(candidate, host, target_log):
    tree = ast.parse(candidate)
    bindings = [node.value for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == '_PR_GRAPH_HOST'
                        for target in node.targets)]
    if len(bindings) != 1 or ast.literal_eval(bindings[0]) != host:
        raise ValueError('candidate/translation-unit identity mismatch')
    modules = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == '_pr_graph_module']
    if len(modules) != 1:
        raise ValueError('ambiguous host module')
    calls = [node for node in ast.walk(modules[0]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'load_inline']
    if len(calls) != 1:
        raise ValueError('ambiguous extension build')
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    flags = ast.literal_eval(keywords['extra_cflags'])
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        raise ValueError('nonliteral host flags')
    requirement = re.search(r'C\+\+(\d+) or later compatible compiler is required', target_log)
    if requirement is None:
        raise ValueError('missing target header requirement')
    build_lines = [line for line in target_log.splitlines()
                   if ' -c ' in line and '-DTORCH_EXTENSION_NAME=' in line]
    if not build_lines:
        raise ValueError('missing target compiler invocation')
    target_flags = re.findall(r'-std=[A-Za-z0-9+]+', build_lines[0])
    if not target_flags:
        raise ValueError('missing target default standard')
    default_flag = target_flags[0]
    effective = standard([default_flag] + flags)
    minimum = int(requirement.group(1))
    return dict(extra_cflags=flags, target_default_standard=default_flag,
                observed_target_effective_standard=standard(target_flags),
                candidate_effective_standard=effective, target_minimum_standard=minimum,
                target_language_contract_pass=effective >= minimum,
                target_environment_reproduced=False, target_abi_proved=False)
