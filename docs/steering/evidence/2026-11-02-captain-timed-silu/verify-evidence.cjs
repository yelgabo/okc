const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const zlib = require('node:zlib');
const childProcess = require('node:child_process');

const root = path.resolve(__dirname, '../../../..');
const read = name => JSON.parse(fs.readFileSync(path.join(__dirname, name), 'utf8'));
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const boards = read('boards.json').map(record => {
    const compressed = Buffer.from(fs.readFileSync(path.join(__dirname, record.file), 'utf8'), 'base64');
    assert.equal(digest(compressed), record.gzip_sha256);
    const raw = zlib.gunzipSync(compressed);
    assert.equal(raw.length, record.raw_bytes);
    assert.equal(digest(raw), record.raw_sha256);
    return JSON.parse(raw);
});
const survey = read('survey.json');
const budget = read('conditional-budget.json');
assert.deepEqual(boards[0].rows.find(row => row.team === 'yelgabo'), survey.public_rank.overall);
assert.equal(boards[0].rows.length, survey.public_rank.overall_rows);
assert.equal(boards[1].rows.length, survey.public_rank.layer_rows);
assert.equal(boards[1].rows.findIndex(row => row.team === 'yelgabo') + 1, survey.public_rank.layer_rank);
assert.deepEqual(boards[1].rows[99], survey.public_rank.layer100);
assert.equal(survey.jacob.layer_profile_count, 37);
assert.equal(survey.jacob.profile_manifest.length, 37);
assert.equal(survey.jacob.minimum_cells.length, 12);
assert(survey.jacob.minimum_cells.every(cell => cell.min_jacob_us > cell.ours_us));
const profile = JSON.parse(childProcess.execFileSync('git', [
    'show', 'HEAD:muse_decode_layer/profiles/g1-stage-release.607eb52bdaf24160.json'
], { cwd: root, encoding: 'utf8', maxBuffer: 4000000 }));
assert.equal(profile.job_id, budget.incumbent_job);
assert.equal(profile.benchmarks.length, 12);
assert.equal(profile.profiles.length, 12);
const loads = [...new Set(profile.benchmarks.map(cell => cell.case.load))];
assert.equal(loads.length, 6);
function factor(fraction, overhead) {
    const gains = loads.map(load => {
        const cells = profile.benchmarks.filter(cell => cell.case.load === load);
        assert.equal(cells.length, 2);
        let before = 0;
        let after = 0;
        for (const cell of cells) {
            assert.equal(cell.case.weight, cell.case.layer === 'sliding' ? 46 : 16);
            const report = profile.profiles.find(record =>
                record.case.load === load && record.case.layer === cell.case.layer).report;
            const row = report.split('\n').find(line => line.includes('moe_fused'));
            const expertUs = Number(row.match(/\s([\d.]+)\s*µs/)[1]);
            const affected = budget.affected_loads.includes(load);
            before += cell.case.weight * cell.median_ns / 1000;
            after += cell.case.weight * (cell.median_ns / 1000 -
                (affected ? fraction * expertUs : 0) + (affected ? overhead : 0));
        }
        return before / after;
    });
    return Math.exp(gains.reduce((total, gain) => total + Math.log(gain), 0) / gains.length);
}
for (const scenario of budget.conditional_scenarios) {
    assert(Math.abs(factor(scenario.entire_expert_reduction, 0) -
        scenario.whole_score_factor_no_replacement) < 1e-12);
}
for (const scenario of budget.required_scenarios) {
    assert(Math.abs(factor(scenario.expert_reduction_for_one_percent_score,
        scenario.assumed_extra_us_per_affected_cell) - 1.01) < 1e-12);
    assert(Math.abs(factor(scenario.expert_reduction_for_current_layer100_point_ratio,
        scenario.assumed_extra_us_per_affected_cell) - survey.public_rank.layer100_ratio) < 1e-12);
}
assert.equal(budget.table_entries * budget.entry_bytes, budget.table_bytes);
assert.equal(budget.table_entries / budget.nonlinear_per_local_route_token,
    budget.entry_count_break_even_local_routes);
process.stdout.write('PASS raw boards, exact standings, 37-profile ledger and conditional weighted-budget recomputation\n');
