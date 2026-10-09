// Source is supplied by launch_training.py; no hard-coded path or process name.
'use strict';
const lua = Process.getModuleByName('lua51.dll');
const fn = (name, result, args) => new NativeFunction(lua.getExportByName(name), result, args);
const load = fn('luaL_loadstring', 'int', ['pointer', 'pointer']);
const call = fn('lua_pcall', 'int', ['pointer', 'int', 'int', 'int']);
const top = fn('lua_gettop', 'int', ['pointer']);
const settop = fn('lua_settop', 'void', ['pointer', 'int']);
const string = fn('lua_tolstring', 'pointer', ['pointer', 'int', 'pointer']);
const source = Memory.allocUtf8String(BOOTSTRAP_SOURCE);
let busy = false, done = false, listener;
const depths = new Map();
const counters = new Map();
function tick(L) {
    if (busy || done) return;
    busy = true;
    let saved;
    try {
        saved = top(L);
        let rc = load(L, source);
        if (!rc) rc = call(L, 0, 1, 0);
        const ptr = string(L, -1, NULL);
        const result = ptr.isNull() ? '' : ptr.readUtf8String();
        if (rc || result === 'READY') {
            done = true;
            send({status: rc ? 'error' : 'ready', detail: result, hook: 'outer_lua_pcall_leave'});
            setTimeout(() => listener.detach(), 0);
        }
    } catch (error) {
        done = true;
        send({status: 'error', detail: String(error), hook: 'outer_lua_pcall_leave'});
        setTimeout(() => listener.detach(), 0);
    } finally {
        if (saved !== undefined) settop(L, saved);
        busy = false;
    }
}
// Never enter Lua from a pushnumber callback while its VM is executing.
// Track nested pcall calls per thread and bootstrap only after the outermost
// successful protected call returns. This startup hook detaches when ready.
listener = Interceptor.attach(lua.getExportByName('lua_pcall'), {
    onEnter(args) {
        this.L = args[0];
        const depth = (depths.get(this.threadId) || 0) + 1;
        depths.set(this.threadId, depth);
        this.outer = depth === 1;
    },
    onLeave(result) {
        const depth = (depths.get(this.threadId) || 1) - 1;
        if (depth) depths.set(this.threadId, depth);
        else depths.delete(this.threadId);
        if (!this.outer || done || busy || result.toInt32() !== 0) return;
        const n = (counters.get(this.threadId) || 0) + 1;
        counters.set(this.threadId, n);
        if (n === 1 || (n & 15) === 0) tick(this.L);
    }
});
