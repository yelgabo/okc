const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const crypto = require('node:crypto');
const prior = path.resolve(__dirname, '../2026-11-02-worker-g224-reference-readiness');
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const source = zlib.gunzipSync(fs.readFileSync(path.join(prior, 'original-selected.cu.gz')));
assert.equal(digest(source), 'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71');
const sourceLines = source.toString().split('\n');
assert.ok(sourceLines[1973].includes('fork_bf16(d4[h] * gsc)'));
assert.ok(sourceLines[1974].includes('const float v = SILU_DIV(gt'));
const expected = ['mul.f32', 'ex2.approx.f32', 'add.f32', 'div.approx.f32', 'mul.f32'];
function extract(text) {
  const lines = text.split('\n'), blocks = [];
  let entry = null;
  for (let index = 0; index < lines.length; index++) {
    const match = lines[index].match(/\.entry\s+([^\s(]+)/);
    if (match) entry = match[1];
    if (!/^\s*\.loc\s+1\s+1975\b/.test(lines[index])) continue;
    const instructions = [];
    for (let next = index + 1; next < lines.length && !/^\s*\.loc\b/.test(lines[next]); next++) {
      const line = lines[next].trim();
      if (line && !line.startsWith('//')) instructions.push(line);
    }
    blocks.push({entry, ptx_line: index + 1, instructions});
  }
  return blocks;
}
assert.equal(extract(['.entry example(', '.loc 1 19750 9', 'ex2.approx.f32 x,y;'].join('\n')).length, 0);
assert.equal(extract(['.entry example(', '.loc 2 1975 9', 'ex2.approx.f32 x,y;'].join('\n')).length, 0);
const result = {source_sha256: digest(source), source_line: 1975, scope: 'Archived original secondary PTX only; no new compile, SASS census, runtime or timing.', families: []};
for (const family of ['B4', 'MAIN']) {
  const ptx = zlib.gunzipSync(fs.readFileSync(path.join(prior, family + '-device.ptx.gz')));
  const manifest = JSON.parse(fs.readFileSync(path.join(prior, family + '-manifest.json')));
  assert.equal(digest(ptx), manifest.ptx_sha256);
  const blocks = extract(ptx.toString()), perEntry = {};
  assert.equal(blocks.length, 216);
  for (const block of blocks) {
    assert.ok(block.entry && /moe_fusedILb[01]E/.test(block.entry));
    assert.deepEqual(block.instructions.map(line => line.split(/\s+/)[0]), expected);
    assert.ok(block.instructions[0].includes('0fBFB8AA3B'));
    assert.ok(block.instructions[2].includes('0f3F800000'));
    perEntry[block.entry] = (perEntry[block.entry] || 0) + 1;
  }
  assert.deepEqual(Object.values(perEntry), [108, 108]);
  result.families.push({family, ptx_sha256: digest(ptx), static_copies: blocks.length, per_entry: perEntry, opcodes: expected, example: blocks[0]});
}
process.stdout.write(JSON.stringify(result, null, 2) + '\n');

