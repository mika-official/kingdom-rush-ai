"""Python client for the injected Lua service (127.0.0.1:9001).

    from kr_client import KRClient
    c = KRClient()
    c.load_env()                  # sends kr_env.lua to the game; no re-injection needed
    obs = c.obs()
    res = c.turn([("build", 3, "archer")], ticks=30)

Only ONE client may be connected at a time.
"""
import json
import os
import socket
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRUNC_MARK = "...(\u622a\u65ad)"   # boot.lua appends this when a reply is cut off


def lua_string(s):
    """Encode s as a single-line, pure-ASCII Lua string literal (decimal escapes)."""
    out = ['"']
    for b in s.encode("utf-8"):
        if b in (34, 92):
            out.append("\\" + chr(b))
        elif 32 <= b < 127:
            out.append(chr(b))
        else:
            out.append("\\%03d" % b)
    out.append('"')
    return "".join(out)


def lua_literal(v):
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return lua_string(v)
    if isinstance(v, (list, tuple)):
        return "{" + ",".join(lua_literal(x) for x in v) + "}"
    if isinstance(v, dict):
        return "{" + ",".join("[%s]=%s" % (lua_literal(k), lua_literal(x)) for k, x in v.items()) + "}"
    raise TypeError("cannot convert %r to Lua" % (type(v),))


class KRClient:
    def __init__(self, host="127.0.0.1", port=9001, timeout=60):
        self.host,self.port=host,port
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.buf = b""
        self.failed = False

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- raw ----
    def raw(self, line, unescape=True):
        try:
            return self._raw(line, unescape)
        except TimeoutError as exc:
            self.failed=True
            raise TimeoutError('Game reply timeout on %s:%s; received %d bytes without a complete line; command=%s' %
                               (self.host,self.port,len(self.buf),line[:160])) from exc
        except (OSError, ConnectionError):
            self.failed = True
            raise

    def _raw(self, line, unescape=True):
        self.sock.sendall((line.strip() + "\n").encode("utf-8"))
        while b"\n" not in self.buf:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise ConnectionError("connection closed by the game")
            self.buf += chunk
        data, _, self.buf = self.buf.partition(b"\n")
        text = data.decode("utf-8", "replace")
        return text.replace("\\n", "\n") if unescape else text

    def call(self, expr):
        """Evaluate a Lua expression that returns a JSON string; return the parsed value."""
        text = self.raw(expr, unescape=False)
        if text.endswith(TRUNC_MARK):
            raise RuntimeError("reply truncated: restart with the latest boot.lua (reply limit: 2000000 bytes)")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise RuntimeError("Lua did not return JSON: " + text[:400])

    # ---- env ----
    def load_env(self, path=None):
        path = path or os.path.join(HERE, "kr_env.lua")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        with open(os.path.join(HERE, 'kr_strategy.lua'), encoding='utf-8') as f:
            src = 'local Strategy = (function()\n' + f.read() + '\nend)()\n' + src
        with open(os.path.join(HERE, 'route_events.lua'), encoding='utf-8') as f:
            src = 'local RouteEvents = (function()\n' + f.read() + '\nend)()\n' + src
        with open(os.path.join(HERE, 'tower_damage_audit.lua'), encoding='utf-8') as f:
            src = 'local DamageAudit = (function()\n' + f.read() + '\nend)()\n' + src
        code = (
            '(function() local f, e = loadstring(%s, "=kr_env"); '
            'if not f then return "LOAD ERROR: " .. tostring(e) end; '
            'local ok, r = pcall(f); '
            'if not ok then return "RUN ERROR: " .. tostring(r) end; '
            'if AI and AI.training_attach then AI.training_attach() end; '
            'return "OK" end)()' % lua_string(src)
        )
        res = self.raw(code)
        if res != "OK":
            raise RuntimeError(res)
        return self.call("ENV.info()")

    def enable_multilevel(self, backup_path):
        """Install memory-only storage and export the original slot before transitions."""
        from pathlib import Path
        rng_source = Path(HERE, "training_rng.lua").read_text(encoding="utf-8")
        if self.raw('(function() if AI.rng_isolation_version then return "OK" end; return assert(loadstring(%s))() end)()' % lua_string(rng_source)) != "OK":
            raise RuntimeError('Training RNG isolation failed')
        source = Path(HERE, "training_bridge.lua").read_text(encoding="utf-8")
        reply = self.raw("assert(loadstring(%s))()" % lua_string(source))
        if reply != "OK":
            raise RuntimeError(reply)
        snapshot = self.call("AI.training_snapshot()")
        path = Path(backup_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2, sort_keys=True)
        import hashlib
        self.snapshot_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def switch_level(self, level, mode=1, difficulty=2, timeout=90):
        self.call("AI.switch_level(%d,%d,%d)" % (level, mode, difficulty))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.call("AI.switch_status()")
            if status.get("ready"):
                info = self.load_env()
                if (info.get("level_idx"), info.get("level_mode"), info.get("difficulty")) != (level, mode, difficulty):
                    raise RuntimeError("Loaded level does not match request: %r" % info)
                return info
            time.sleep(0.05)
        raise TimeoutError("Level transition timed out: level=%s mode=%s" % (level, mode))

    def obs(self):
        return self.call("ENV.obs_json()")

    def turn(self, actions=(), ticks=1):
        return self.call("ENV.turn(%s,%d)" % (lua_literal([list(a) for a in actions]), ticks))

    def manual(self, on=True):
        if self.failed:
            if not on:
                return  # Do not wait another socket timeout while unwinding a failed episode.
            raise ConnectionError("Game connection failed")
        self.raw("ENV.manual(%s)" % ("true" if on else "false"))

    def reset(self, settle_ticks=10):
        self.raw("ENV.reset()")
        return self.turn([], settle_ticks)["obs"]
