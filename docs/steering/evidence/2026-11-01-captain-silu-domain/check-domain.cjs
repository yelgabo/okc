const assert = require('node:assert/strict');
const storage = new ArrayBuffer(4);
const words = new Uint32Array(storage);
const values = new Float32Array(storage);
const counts = {raw_words:65536,finite:0,infinity:0,nan:0,subnormal:0,finite_roundtrip_failures:0,finite_index_failures:0,negative_zero_preserved:false};
for (let rawWord=0; rawWord<65536; rawWord+=1) {
  const exponent = (rawWord >>> 7) & 255;
  const fraction = rawWord & 127;
  words[0] = (rawWord << 16) >>> 0;
  if (exponent===255) {
    counts[fraction===0 ? 'infinity' : 'nan']+=1;
    continue;
  }
  counts.finite+=1;
  if (exponent===0 && fraction!==0) counts.subnormal+=1;
  const originalBits=words[0];
  const floatValue=values[0];
  values[0]=Math.fround(floatValue);
  if (words[0]!==originalBits) counts.finite_roundtrip_failures+=1;
  if ((words[0]>>>16)!==rawWord) counts.finite_index_failures+=1;
  if (rawWord===32768) counts.negative_zero_preserved=Object.is(floatValue,-0);
}
assert.equal(counts.finite,65280);
assert.equal(counts.infinity,2);
assert.equal(counts.nan,254);
assert.equal(counts.subnormal,254);
assert.equal(counts.finite_roundtrip_failures,0);
assert.equal(counts.finite_index_failures,0);
assert.equal(counts.negative_zero_preserved,true);
console.log(JSON.stringify({scope:'CPU IEEE-754 embedding and index only; not CUDA intrinsic, NaN payload, epilogue equivalence, legality or performance proof',counts},null,2));

