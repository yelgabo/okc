import hashlib
import json
from pathlib import Path

HEADER = Path('/root/.cache/okc-cuda130-88/nvidia/cu13/include/cuda_occupancy.h')
raw = HEADER.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '163b8f296ef2a33c799e2dc8c525b37166da9217bcc6a3c1d26b1c10aca5e22b'
text = raw.decode()
for equation in (
    'regsPerWarp          = attributes->numRegs * properties->warpSize;',
    'regsAllocatedPerWarp = __occRoundUp(regsPerWarp, allocationGranularity);',
    'numRegsPerSubPartition  = properties->regsPerMultiprocessor / numSubPartitions;',
    'numWarpsPerSubPartition = numRegsPerSubPartition / regsAllocatedPerWarp;',
    'numWarpsPerSM = numWarpsPerSubPartition * numSubPartitions;',
    'maxBlocks     = numWarpsPerSM / warpsAllocatedPerCTA;',
):
    assert equation in text, equation
register_granularity = text.split('cudaOccRegAllocationGranularity(int', 1)[1].split('return CUDA_OCC_SUCCESS;', 1)[0]
subpartitions = text.split('cudaOccSubPartitionsPerMultiprocessor(int', 1)[1].split('return CUDA_OCC_SUCCESS;', 1)[0]
assert 'case 9:' in register_granularity and 'value = 256;' in register_granularity
assert 'case 9:' in subpartitions and 'value = 4;' in subpartitions


def model(registers):
    allocated_per_warp = ((registers * 32 + 255) // 256) * 256
    per_partition_warps = 16384 // allocated_per_warp
    assumed_per_block = allocated_per_warp * 12
    block_limit = 0 if assumed_per_block > 65536 else (per_partition_warps * 4) // 10
    return dict(registers=registers, allocated_per_warp=allocated_per_warp,
                warps_per_partition=per_partition_warps, register_only_block_limit=block_limit)


ceiling = max(registers for registers in range(1, 256) if model(registers)['register_only_block_limit'] >= 2)
assert ceiling == 96
assert model(96)['register_only_block_limit'] == 2
assert model(97)['register_only_block_limit'] == 1
assert model(102)['register_only_block_limit'] == 1
print(json.dumps(dict(scope='Pinned header-equation CPU model, NOT native occupancy execution or target measurement',
    header_sha256=hashlib.sha256(raw).hexdigest(), threads_per_block=320, warps_per_block=10,
    assumptions=dict(registers_per_sm=65536, registers_per_block=65536, warp_size=32,
                     register_allocation_granularity=256, subpartitions=4, partitioned_gc='off'),
    two_cta_necessary_uniform_register_ceiling=ceiling,
    unrounded_incorrect_ceiling=65536 // (2 * 320),
    rows=[model(registers) for registers in (80, 88, 96, 97, 100, 102, 104, 128)],
    target_occupancy=None), indent=2))
