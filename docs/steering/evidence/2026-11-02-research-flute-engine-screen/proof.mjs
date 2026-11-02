import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const buffer = new ArrayBuffer(4);
const view = new DataView(buffer);
function bf16(value) {
  view.setFloat32(0, value, false);
  const bits = view.getUint32(0, false);
  const rounded = ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) & 0xffff;
  view.setUint32(0, rounded << 16, false);
  return view.getFloat32(0, false);
}
const codes = [0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6];
const scales = [1, 1.125, 1.25, 1.375];
const products = scales.flatMap(scale => codes.map(code => bf16(code * scale)));
assert.equal(new Set(products).size, 57);
assert(scales.every(scale => codes.every(code => bf16(code * scale) === code * scale)));
assert.equal(bf16(bf16(257) + bf16(-256)), 0);
assert.equal(257 - 256, 1);
assert.equal(bf16(257 * 1.125), 290);
assert.equal(bf16(bf16(257) * 1.125), 288);
const control = await readFile('muse_decode_layer/cand/g1b4_stage_release.py');
assert.equal(createHash('sha256').update(control).digest('hex'), 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab');
const prior = JSON.parse(await readFile('docs/steering/evidence/2026-11-02-research-broad-expert-feasibility/analysis.json', 'utf8'));
const loads = ['decode 16', 'decode 32', 'verify 16', 'verify 32'];
function gain(fraction, overhead) {
  return Math.exp(loads.reduce((total, load) => total + Math.log(prior.control_us[load] / (prior.control_us[load] - fraction * prior.expert_activity_us[load] + overhead)), 0) / 6) - 1;
}
const budgets = [0.03, 0.10].map(target => ({ target, scenarios: [0, 5, 10].map(overhead => {
  let lower = 0;
  let upper = .9;
  for (let iteration = 0; iteration < 80; iteration++) {
    const midpoint = (lower + upper) / 2;
    if (gain(midpoint, overhead) < target) lower = midpoint;
    else upper = midpoint;
  }
  const fraction = (lower + upper) / 2;
  assert(Math.abs(gain(fraction, overhead) - target) < 1e-12);
  const previous = prior.requirements.find(row => Math.abs(row.target - target) < 1e-12).scenarios.find(row => row.unmeasured_net_added_cost_us === overhead);
  assert(Math.abs(fraction - previous.conditional_gross_activity_fraction) < 1e-12);
  return { overhead_us_unmeasured: overhead, required_fully_exposed_activity_fraction: fraction };
}) }));
console.log(JSON.stringify({ source_sha256: prior.source_sha256, group64_unique_values: new Set(products).size, group64_single_table_capacity: 16, partial_rounding: { fp32: 1, mixed_bf16: 0 }, global_postscale: { correct_boundary: 290, early_bf16: 288 }, budgets, scope: 'Own CPU arithmetic and source hash only; no donor import/execution or target APIs.' }, null, 2));
