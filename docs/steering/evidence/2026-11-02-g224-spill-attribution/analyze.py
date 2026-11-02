import collections
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
ARCHIVE = ROOT / 'docs/steering/evidence/2026-11-01-worker-g224-prefix-resource-stop'
SOURCE = ROOT / 'docs/steering/evidence/2026-11-01-worker-g2-24-route-prefix/selected.cu'
native = SOURCE.read_text()
assert hashlib.sha256(native.encode()).hexdigest() == 'ed082519a4312bc1012192134b2d2865016789802b0454381984260c7aa5ca6f'
device = native[:native.index('#include <torch/extension.h>')]
for assertion in (
    'static_assert(G224_OFFSETOF(Smem, wc) == G224_OFFSETOF(Smem, wcp), "G224 alias offset");',
    'static_assert(G224_OFFSETOF(Smem, wcpB) >= G224_OFFSETOF(Smem, wc) + sizeof(((Smem*)0)->wc), "G224 separate prefix B");',
):
    assert device.count(assertion) == 1
    device = device.replace(assertion, '')
anchor = '}  // namespace okc'
probes = {'wc': 'G224_OFFSETOF(Smem, wc)', 'wcp': 'G224_OFFSETOF(Smem, wcp)', 'wcpB': 'G224_OFFSETOF(Smem, wcpB)', 'sizeofwc': 'sizeof(((Smem*)0)->wc)', 'sizeofsmem': 'sizeof(Smem)'}
functions = '\n'.join('extern "C" __device__ __noinline__ unsigned long long g224_probe_'+name+'() { unsigned long long result; asm volatile("mov.u64 %0, %1;" : "=l"(result) : "n"('+value+')); return result; }' for name,value in probes.items())
functions += '\nextern "C" __global__ void g224_layout_anchor(unsigned long long* output) { ' + ' '.join('output['+str(index)+']=g224_probe_'+name+'();' for index,name in enumerate(probes)) + ' }'
device = device.replace(anchor, functions + '\n' + anchor)
decl = '__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n'
device = device.replace('template <bool PB>\n__device__ __forceinline__ void wgs_phase(', decl + 'template <bool PB>\n__device__ __forceinline__ void wgs_phase(')
source_lines = native.splitlines()
device_lines = device.splitlines()
line_map = {}
for tag, first, last, second, end in difflib.SequenceMatcher(None, device_lines, source_lines, autojunk=False).get_opcodes():
    if tag == 'equal':
        line_map.update({first + offset + 1: second + offset + 1 for offset in range(last-first)})


def parse(text):
    functions = collections.defaultdict(list)
    symbol = None
    locations = []
    last_location = False
    for line in text.splitlines():
        match = re.search(r'\.section\s+\.text\.([^,]+),', line)
        if match:
            symbol = match[1]
            locations = []
        if '//## File' in line:
            if not last_location:
                locations = []
            locations.append(line.strip())
            last_location = True
            continue
        match = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;', line)
        if match and symbol:
            last_location = False
            if match[2].startswith('.'):
                continue
            functions[symbol].append({'pc': int(match[1], 16), 'instruction': match[2], 'debug': locations[:]})
    return functions


def graph(dot, symbol):
    marker = 'subgraph "cluster_' + symbol + '" {'
    fragment = dot.split(marker, 1)[1].split('subgraph "cluster_', 1)[0]
    nodes = {}
    for match in re.finditer(r'^"([^"]+)"\s*\n\[label="(.*?)"\]', fragment, re.M | re.S):
        pcs = [int(value, 16) for value in re.findall(r'(?:\\l|>)([0-9a-f]+):', match[2])]
        nodes[match[1]] = pcs
    edges = re.findall(r'^"([^"]+)"[^\n]*? -> "([^"]+)"', fragment, re.M)
    successors = {node: set() for node in nodes}
    predecessors = {node: set() for node in nodes}
    for first, second in edges:
        if first in nodes and second in nodes:
            successors[first].add(second)
            predecessors[second].add(first)
    entry = '.text.' + symbol
    reachable = set()
    pending = [entry]
    while pending:
        node = pending.pop()
        if node not in reachable:
            reachable.add(node)
            pending.extend(successors[node])
    dominators = {node: ({entry} if node == entry else set(reachable)) for node in reachable}
    changed = True
    while changed:
        changed = False
        for node in reachable - {entry}:
            parents = predecessors[node] & reachable
            updated = {node} | set.intersection(*(dominators[parent] for parent in parents))
            if updated != dominators[node]:
                dominators[node] = updated
                changed = True
    loops = []
    for tail, header in edges:
        if tail not in reachable or header not in dominators[tail]:
            continue
        members = {header, tail}
        pending = [] if tail == header else [tail]
        while pending:
            node = pending.pop()
            for parent in predecessors[node] & reachable:
                if parent not in members:
                    members.add(parent)
                    pending.append(parent)
        pcs = sorted(pc for node in members for pc in nodes[node])
        loops.append({'header': header, 'tail': tail, 'nodes': sorted(members), 'pcs': pcs})
    return nodes, loops, reachable


result = {'source_sha256': hashlib.sha256(native.encode()).hexdigest(), 'adapted_device_sha256': hashlib.sha256(device.encode()).hexdigest(), 'families': {}}
table = ['family\tentry\tpc\topcode_operands\tdebug_inline_hint\texecution_region\tnative_lines\tptx_line\tptx_text\tinnermost_loop_header\tloop_has_HGMMA']
for family in ('B4', 'MAIN'):
    manifest = json.loads((ARCHIVE / (family+'-manifest.json')).read_text())
    assert result['adapted_device_sha256'] == manifest['compiled_device_prefix_sha256']
    ptx = gzip.decompress((ARCHIVE / (family+'-device.ptx.gz')).read_bytes()).decode()
    assert hashlib.sha256(ptx.encode()).hexdigest() == manifest['ptx_sha256']
    ptx_lines = ptx.splitlines()
    source_sass = parse(gzip.decompress((OUT / (family+'-source.sass.gz')).read_bytes()).decode())
    ptx_sass = parse(gzip.decompress((OUT / (family+'-ptx.sass.gz')).read_bytes()).decode())
    dot = gzip.decompress((OUT / (family+'-cfg.dot.gz')).read_bytes()).decode()
    record = {'ptx_local_declarations_or_operations': re.findall(r'^.*(?:\.local|ld\.local|st\.local).*$', ptx, re.M), 'entries': {}}
    for symbol, instructions in source_sass.items():
        if 'moe_fused' not in symbol:
            continue
        nodes, loops, reachable = graph(dot, symbol)
        pc_nodes = {pc: node for node, pcs in nodes.items() for pc in pcs}
        instruction_map = {row['pc']: row for row in instructions}
        ptx_map = {row['pc']: row for row in ptx_sass[symbol]}
        rows = []
        for row in instructions:
            if not re.search(r'\b(?:LDL|STL)(?:\.|\s)', row['instruction']):
                continue
            locations = [int(value) for note in row['debug'] for value in re.findall(r'line (\d+)', note)]
            native_locations = [line_map.get(value) for value in locations]
            if 2569 in native_locations:
                role = 'activation_gather'
            elif 2576 in native_locations:
                role = 'phase_A_segment_control'
            elif 2578 in native_locations:
                role = 'phase_B_segment_control_or_epilogue'
            else:
                role = 'entry_other'
            matches = [loop for loop in loops if row['pc'] in loop['pcs']]
            matches.sort(key=lambda loop: len(loop['pcs']))
            loop_info = []
            for loop in matches:
                loop_info.append({'header': loop['header'], 'tail': loop['tail'], 'instruction_count': len(loop['pcs']), 'min_pc': hex(min(loop['pcs'])), 'max_pc': hex(max(loop['pcs'])), 'contains_HGMMA': any('HGMMA' in instruction_map.get(pc, {}).get('instruction', '') for pc in loop['pcs'])})
            ptx_debug = ptx_map[row['pc']]['debug']
            ptx_line = int(re.search(r'line (\d+)', ptx_debug[0])[1]) if ptx_debug else None
            gather_header_pc = 0x4660
            phase_header_pc = 0x17be0 if family == 'B4' else 0x17e10
            if any(int(loop['min_pc'], 16) == gather_header_pc and not loop['contains_HGMMA'] for loop in loop_info):
                execution_region = 'gather_batch_loop'
            elif any(int(loop['min_pc'], 16) == phase_header_pc and loop['contains_HGMMA'] for loop in loop_info):
                execution_region = 'phase_B_outer_segment_loop_not_inner_stage_loop'
            else:
                execution_region = 'phase_B_entry_or_final_drain'
            access = dict(row, native_lines=native_locations, native_text=[source_lines[value-1] if value else None for value in native_locations], role=role, execution_region=execution_region, ptx_line=ptx_line, ptx_text=ptx_lines[ptx_line-1] if ptx_line else None, cfg_node=pc_nodes.get(row['pc']), reachable=pc_nodes.get(row['pc']) in reachable, natural_loops=loop_info)
            rows.append(access)
            table.append('\t'.join([family, symbol, hex(row['pc']), row['instruction'], role, execution_region, ','.join(str(value) for value in native_locations), str(ptx_line), str(access['ptx_text']).strip(), loop_info[0]['header'] if loop_info else '-', str(loop_info[0]['contains_HGMMA']) if loop_info else '-']))
        counts = collections.Counter(row['role'] for row in rows)
        widths = collections.Counter()
        for row in rows:
            operation = row['instruction'].split()[0]
            width = 16 if '.128' in operation else 8 if '.64' in operation else 4
            widths[('load' if operation.startswith('LDL') else 'store') + '_static_operand_bytes'] += width
        loop_summary = []
        for loop in loops:
            if not any('HGMMA' in instruction_map.get(pc, {}).get('instruction', '') for pc in loop['pcs']):
                continue
            loop_summary.append({'header': loop['header'], 'tail': loop['tail'], 'min_pc': hex(min(loop['pcs'])), 'max_pc': hex(max(loop['pcs'])), 'instruction_count': len(loop['pcs']), 'local_access_pcs': [hex(row['pc']) for row in rows if row['pc'] in loop['pcs']]})
        record['entries'][symbol] = {'counts_debug_inline_hint_only': dict(counts), 'counts_execution_region': dict(collections.Counter(row['execution_region'] for row in rows)), 'static_instruction_encoding_bytes': 16*len(rows), 'static_operand_width_sum_not_executed_traffic': dict(widths), 'rows': rows, 'natural_loop_count': len(loops), 'hgmma_natural_loops': loop_summary, 'cfg_pc_coverage': len(pc_nodes), 'instruction_count': len(instructions)}
        print(family, symbol, dict(counts), dict(widths), 'loops',len(loops), 'coverage',len(pc_nodes),len(instructions))
    record['all_text_local_access_counts'] = {symbol: sum(bool(re.search(r'\b(?:LDL|STL)(?:\.|\s)', row['instruction'])) for row in stream) for symbol, stream in source_sass.items()}
    assert record['ptx_local_declarations_or_operations'] == []
    entries = list(record['entries'].values())
    assert len(entries) == 2
    assert [(row['pc'], row['instruction'], row['execution_region']) for row in entries[0]['rows']] == [(row['pc'], row['instruction'], row['execution_region']) for row in entries[1]['rows']]
    for symbol, entry in record['entries'].items():
        assert all(row['reachable'] for row in entry['rows'])
        widths = entry['static_operand_width_sum_not_executed_traffic']
        assert widths['store_static_operand_bytes'] == manifest['resources'][symbol]['spill_store_bytes']
        assert widths['load_static_operand_bytes'] == manifest['resources'][symbol]['spill_load_bytes']
        assert len(entry['hgmma_natural_loops']) == 8
        assert sum(bool(loop['local_access_pcs']) for loop in entry['hgmma_natural_loops']) == 1
    record['validation'] = 'PASS: both-bool local PC/opcode/region identity, all local PCs CFG-reachable, complete static operand-width match to assembler spill rows, no PTX local operations, seven of eight HGMMA-containing natural loops have no local accesses'
    result['families'][family] = record
(OUT / 'attribution.json').write_text(json.dumps(result, indent=2)+'\n')
(OUT / 'local-accesses.tsv').write_text('\n'.join(table)+'\n')
print('Bound every local PC to source, archived PTX debug and emitted CFG natural loops')
