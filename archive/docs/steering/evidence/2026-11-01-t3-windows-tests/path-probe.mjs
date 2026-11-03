import * as Effect from './node_modules/effect/dist/Effect.js';
import * as Path from './node_modules/effect/dist/Path.js';
import * as NodePath from './node_modules/@effect/platform-node/dist/NodePath.js';
const windows = String.raw`C:\Users\test\AppData\Local\Temp\t3-start\project\worktree`;
const posix = '/tmp/provider-turn-start-worktrees/project/restore';
const results = [];
for (const [name, layer] of [['effect-posix', Path.layer], ['node-native', NodePath.layer], ['node-win32', NodePath.layerWin32], ['node-posix', NodePath.layerPosix]]) {
  const path = Effect.runSync(Effect.provide(Path.Path, layer));
  for (const input of [windows, posix]) {
    results.push({service:name,input,separator:path.sep,isAbsolute:path.isAbsolute(input),resolved:path.resolve(input),normalized:path.resolve(input) === input,leaseKeyMatchesCleanup:path.resolve(input) === input});
  }
}
console.log(JSON.stringify({platform:process.platform,results},null,2));
