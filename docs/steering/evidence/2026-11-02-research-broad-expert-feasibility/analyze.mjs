import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const directory=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(directory,'../../../..');
const read=relative=>fs.readFileSync(path.join(root,relative),'utf8');
const hash=text=>crypto.createHash('sha256').update(text).digest('hex');
const source=read('muse_decode_layer/cand/g1b4_stage_release.py').replaceAll('\r\n','\n');
assert.equal(hash(source),'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab');
const receiptPath='docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json';
const receipt=JSON.parse(read(receiptPath));
assert.equal(receipt.id,'cc474883117947c0');
assert.equal(receipt.result.passed,true);
const control={},expert={},cells=[];
for(const record of receipt.result.records.filter(record=>record.event==='benchmark')){
 const load=record.case.load;
 control[load]=(control[load]??0)+record.case.weight*record.median_ns/62000;
 cells.push({load,context:record.case.layer,weight:record.case.weight,adjusted_us:record.median_ns/1000});
}
for(const record of receipt.result.records.filter(record=>record.event==='profile')){
 const activity=record.report.match(/okc::moe_fused[^\n]*?\s([0-9.]+) µs/);
 assert.ok(activity);
 const load=record.case.load;
 expert[load]=(expert[load]??0)+record.case.weight*Number(activity[1])/62;
}
const loads=['decode 16','decode 32','verify 16','verify 32'];
const gain=(fraction,overhead,selected=loads)=>Math.exp(selected.reduce((total,load)=>total+Math.log(control[load]/(control[load]-fraction*expert[load]+overhead)),0)/6)-1;
const solve=(target,overhead,selected=loads)=>{
 let low=0,high=.99;
 for(let iteration=0;iteration<90;iteration++){
  const middle=(low+high)/2;
  if(gain(middle,overhead,selected)<target)low=middle;else high=middle;
 }
 return (low+high)/2;
};
const requirements=[.03,.06576288172015832,.1].map(target=>({target,scenarios:[0,5,10].map(overhead=>{
 const fraction=solve(target,overhead);
 assert.ok(Math.abs(gain(fraction,overhead)-target)<1e-12);
 return {unmeasured_net_added_cost_us:overhead,conditional_gross_activity_fraction:fraction,gross_us:Object.fromEntries(loads.map(load=>[load,fraction*expert[load]])),net_endpoint_us:Object.fromEntries(loads.map(load=>[load,fraction*expert[load]-overhead]))};
}),main_only_no_cost_fraction:solve(target,0,['verify 16','verify 32'])}));
const p1=JSON.parse(read('docs/steering/evidence/2026-11-02-worker-broad-expert-architecture/analysis.json'));
for(let index=0;index<requirements.length;index++)assert.ok(Math.abs(requirements[index].scenarios[0].conditional_gross_activity_fraction-p1.budgets[index].conditional_expert_activity_fraction)<1e-14);
const manifest=JSON.parse(read('docs/steering/evidence/2026-11-02-worker-broad-expert-architecture/source-manifest.json'));
const inspected=['humming/config/mma.py','humming/schema/humming.py','humming/forward.py','humming/include/humming/arith/epilogue_arith.cuh','humming/include/humming/scheduler.cuh'];
const primary=inspected.map(relative=>{
 const entry=manifest.files.find(entry=>entry.path===relative);
 assert.ok(entry);
 const bytes=fs.readFileSync(path.join(root,'.build/worker-broad-expert-architecture/humming-source',relative+'.txt'));
 assert.equal(hash(bytes),entry.sha256);
 return {path:relative,sha256:entry.sha256,git_blob:entry.git_blob,url:entry.url};
});
const invariants=[
 ['MID split job size',source.includes('const int nsj = (tot + SK_T - 1) / SK_T;')],
 ['MID unpatched helper alias',source.includes("_MA_ALT_SRCS['moe_helper0df'] = _MA_ALT_SRCS['moe_helper0']")],
 ['B4 norm partials',source.includes('p.sspart[(size_t)(jst + tok0) * 256 + r16] = q0;')],
 ['MAIN two-pass finalizer',source.includes('pass 2 re-reads the rows')],
 ['BF16 gamma boundary',source.includes('fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0)))')],
 ['FP32 gamma product',source.includes('#define OKC_YBF 0')],
 ['positive finite PRMT admission',source.includes('((b < 0x08) | (b > 0x7E)).any()')]
];
for(const [name,passed] of invariants)assert.ok(passed,name);
const taskPath='muse_decode_layer/TASK.md';
const task=read(taskPath);
assert.ok(task.includes("NVFP4 weights stay NVFP4 and BF16 weights stay BF16, or smaller, losslessly."));
const storedWeightRule={path:taskPath,commit:'e81d555b0667fd3b9548d72207bfb7a6833acf82',line:102,sha256:hash(task),conclusion:'Design B stores NVFP4-derived weights as BF16 in prepare: not admissible under the explicit stored-precision rule, regardless of operand equivalence.'};
const expertElements=8192*4096+4096*4096;
const units={elements:expertElements,ordinary_bytes:expertElements*9/16,sc6_bytes:expertElements*35/64,bf16_bytes:expertElements*2};
const midExample={description:'Allowed count=16 on one expert: MID has two <=8-token jobs, not one weight pass; no measured count implied',routes:16,experts:1,control_jobs:2,expanded_single_pass_inflation:units.bf16_bytes/(2*units.ordinary_bytes)};
const wgsExample={description:'Allowed count=128 on one expert: control FT_T64 means two jobs; Humming BM16/BM32 token tiles imply8/4 tiles before any physical-cache assumptions',count:128,control_jobs:2,grouped16_tiles:8,grouped32_tiles:4};
const output={source_sha256:hash(source),p1_final_commit:'c8aacece1b6447b577187a7d7fb5bd7d4ea7e90c',p1_report_sha256:hash(read('docs/steering/2026-11-02-worker-broad-expert-architecture.md')),storedWeightRule,receipt:{path:receiptPath,sha256:hash(read(receiptPath)),id:receipt.id},primary,control_us:control,expert_activity_us:expert,cells,requirements,invariants,units,midExample,wgsExample,limits:['All activity-based budgets assume fully exposed activity; overhead0/5/10us is a sensitivity, not measured launch time.','Count examples are legal source distributions, not fresh target routing. Uniform one-pass P1 table is not actual selected route traffic.','No donor import, kernel/harness implementation, compiler or target execution.'],decision:'REJECT: B violates stored-weight precision; A lacks substantial complete-path evidence. No universal bandwidth impossibility claim.'};
fs.writeFileSync(path.join(directory,'analysis.json'),JSON.stringify(output,null,2)+'\n');
console.log(JSON.stringify({checks:invariants.length,matched_p1_budgets:requirements.length,primary_hashes:primary.length,requirements,units,midExample,wgsExample},null,2));
