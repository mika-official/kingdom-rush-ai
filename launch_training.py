"""One command: own game processes -> PID injection -> independent ports -> PPO."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
DEFAULT_GAME = Path("E:/SteamLibrary/steamapps/common/Kingdom Rush/Kingdom Rush.exe")


def check_ports(ports):
    if len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
        raise ValueError("Ports must be unique and in 1..65535")
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(f"Port {port} is unavailable; choose another --base-port") from exc
    finally:
        for sock in sockets:
            sock.close()


def dependency_environment():
    env = os.environ.copy()
    dirs = [str(ROOT / name) for name in (".runtime-deps", ".ppo-test-deps") if (ROOT / name).is_dir()]
    env["PYTHONPATH"] = os.pathsep.join(dirs + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def ensure_dependencies(env, tensorboard=False):
    requirements = {"numpy": "numpy", "gymnasium": "gymnasium", "sb3_contrib": "sb3-contrib", "frida": "frida-tools"}
    if tensorboard:
        requirements["tensorboard"] = "tensorboard"
    missing = []
    for module, package in requirements.items():
        result = subprocess.run([sys.executable, "-c", f"import {module}"], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode:
            missing.append(package)
    if missing:
        print("Installing local dependencies:", ", ".join(missing), flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--target",
                        str(ROOT / ".runtime-deps"), *missing], check=True, env=env)
        env = dependency_environment()
    # Make the same packages available to the launcher's Frida client.
    for directory in reversed(env.get("PYTHONPATH", "").split(os.pathsep)):
        if directory:
            sys.path.insert(0, directory)
    return env


def bootstrap_source(port, slot, difficulty):
    from kr_client import lua_string
    boot = (ROOT / "boot.lua").read_text(encoding="utf-8")
    if boot.count("local PORT = 9001") != 1:
        raise RuntimeError("Unexpected boot.lua port declaration")
    boot = boot.replace("local PORT = 9001", f"local PORT = {port}")
    bridge = (ROOT / "training_bridge.lua").read_text(encoding="utf-8")
    # Only queue work inside Frida's native callback. Run module loading and the
    # screen transition at the next director frame, outside the intercepted call.
    return f'''
local d = rawget(_G, "director") or package.loaded.director
if type(d) ~= "table" or type(d.update) ~= "function" then return "WAIT" end
local original = d.update
d.update = function(self, dt)
    if d.active_item and d.active_item.item_name == "slots"
        and d.active_item.done_callback and not d.active_item.done_callback_called then
        d.update = original
        local storage = require("storage")
        assert(storage:load_slot({slot}), "Selected save slot does not exist; create it once in the game")
        storage.active_slot_idx = {slot}
        assert(loadstring({lua_string(boot)}, "=training_boot"))()
        assert(loadstring({lua_string(bridge)}, "=training_bridge"))()
        AI.switch_level(1, 1, {difficulty})
    end
    return original(self, dt)
end
return "READY"
'''


def inject(device, process, source, timeout, log):
    deadline = time.monotonic() + timeout
    session = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Game PID {process.pid} exited ({process.returncode}); see {log}")
        try:
            session = device.attach(process.pid)
            # DLL loading may lag behind process creation.
            probe = session.create_script("rpc.exports = { ready() { return Process.findModuleByName('lua51.dll') !== null; } };")
            probe.load()
            ready = probe.exports_sync.ready()
            probe.unload()
            if ready:
                break
            session.detach()
            session = None
        except Exception:
            if session:
                session.detach()
                session = None
        time.sleep(0.2)
    if session is None:
        raise TimeoutError("Could not attach to lua51.dll")
    event, state = threading.Event(), {}
    with log.open('a',encoding='utf8') as handle:
        handle.write(json.dumps(dict(type='phase',payload=dict(stage='lua_dll_attached',pid=process.pid)))+'\n')

    def message(msg, data):
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(msg, ensure_ascii=False) + "\n")
        if msg["type"] == "send":
            state.update(msg["payload"])
            event.set()
        elif msg["type"] == "error":
            state.update(status="error", detail=msg.get("stack", str(msg)))
            event.set()

    try:
        js = "const BOOTSTRAP_SOURCE = " + json.dumps(source) + ";\n" + (ROOT / "auto_inject.js").read_text(encoding="utf-8")
        script = session.create_script(js)
        script.on("message", message)
        script.load()
        with log.open('a',encoding='utf8') as handle:
            handle.write(json.dumps(dict(type='phase',payload=dict(stage='outer_pcall_hook_loaded',pid=process.pid)))+'\n')
        while time.monotonic() < deadline and not event.wait(0.2):
            if process.poll() is not None:
                raise RuntimeError(f"Game PID {process.pid} exited during injection (code {process.returncode}); see {log}")
        if not event.is_set():
            raise TimeoutError("Injection timed out waiting for the slot screen; check the game startup screen and logs")
        if state.get("status") != "ready":
            raise RuntimeError("Injection failed: " + state.get("detail", "unknown error"))
        return session, script
    except BaseException:
        try:session.detach()
        except Exception:pass
        raise


def wait_game(port, process, timeout):
    from kr_client import KRClient
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Game PID {process.pid} exited before readiness")
        try:
            with KRClient(port=port, timeout=3) as client:
                while time.monotonic() < deadline:
                    status = client.call("AI.switch_status()")
                    if status.get("ready"):
                        client.load_env()
                        obs = client.turn([], 10)["obs"]
                        if not obs.get("spots"):
                            state = client.raw('{tick=game.store.tick,paused=game.store.paused,entities=table.count(game.store.entities)}')
                            raise RuntimeError(f"Game on port {port} loaded no tower spots: {state}")
                        return
                    time.sleep(0.1)
        except (ConnectionError, OSError):
            time.sleep(0.2)
    raise TimeoutError(f"Game on port {port} did not load its initial level")


def stop_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def start_game(device,args,port,env,directory):
    """Retry only a not-yet-training owned instance; preserve every attempt log."""
    for attempt in range(1,args.startup_attempts+1):
        suffix='' if attempt==1 else f'_attempt_{attempt}'
        output=(directory/f'game_{port}{suffix}.log').open('w',encoding='utf8')
        process=session=None
        try:
            startup=None
            if os.name=='nt':
                startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
            process=subprocess.Popen([str(args.game.resolve()),'-screen','slots','-windowed','-width','800','-height','600'],
                cwd=args.game.resolve().parent,env=env,stdout=output,stderr=subprocess.STDOUT,startupinfo=startup)
            print(f'Game PID={process.pid} port={port}; injecting (attempt {attempt}/{args.startup_attempts})...',flush=True)
            session,script=inject(device,process,bootstrap_source(port,args.slot,args.difficulty),
                args.startup_timeout,directory/f'inject_{port}{suffix}.jsonl')
            wait_game(port,process,args.startup_timeout)
            return process,session,script,output
        except BaseException as exc:
            if session is not None:
                try:session.detach()
                except Exception:pass
            exit_code=process.poll() if process is not None else None
            if process is not None:stop_owned(process)
            output.close()
            with (directory/'startup_attempts.jsonl').open('a',encoding='utf8') as handle:
                handle.write(json.dumps(dict(port=port,attempt=attempt,pid=process.pid if process else None,
                    exit_code=exit_code,error=str(exc)))+'\n')
            if not isinstance(exc,Exception) or attempt==args.startup_attempts:raise
            print(f'Game port={port} startup failed: {exc}; retrying a fresh owned process...',flush=True)
            time.sleep(.5)
def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=9101)
    parser.add_argument("--game", type=Path, default=DEFAULT_GAME)
    parser.add_argument("--slot", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument('--startup-attempts',type=int,default=3)
    parser.add_argument("--eval-every", type=int, default=100000)
    parser.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--modes", default="campaign")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, training_args = parser.parse_known_args()
    if args.workers < 1 or args.slot < 1 or args.startup_timeout <= 0 or args.eval_every < 0 or args.startup_attempts<1:
        parser.error("workers, slot and startup-timeout must be positive; eval-every must be nonnegative")
    for arg in training_args:
        if arg.split("=")[0] in ("--ports", "--port", "--eval-port"):
            parser.error("ports are assigned by this launcher; use --base-port")
    is_eval = any(arg.split("=")[0] in ("--eval", "--baseline") for arg in training_args)
    eval_needed = not args.smoke and (args.eval_every > 0 or is_eval)
    ports = list(range(args.base_port, args.base_port + args.workers))
    all_ports = ports + ([ports[-1] + 1] if eval_needed else [])
    if not args.game.is_file():
        parser.error(f"Game executable not found: {args.game}")
    command = [sys.executable, "-u", str(ROOT / "train_all.py"), "--ports", ",".join(map(str, ports)),
               "--eval-every", str(args.eval_every), "--difficulty", str(args.difficulty)]
    if eval_needed:
        command += ["--eval-port", str(all_ports[-1])]
    if args.smoke:
        command.append("--smoke")
    if is_eval:
        ports = []
        all_ports = [args.base_port]
        command = [sys.executable, "-u", str(ROOT / "train_all.py"), "--port", str(args.base_port),
                   "--eval-every", "0", "--difficulty", str(args.difficulty)]
    command += training_args
    command += ["--modes", args.modes]
    check_ports(all_ports)
    print(json.dumps(dict(game=str(args.game), worker_ports=ports,
                          eval_port=all_ports[-1] if eval_needed else None, command=command), ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    env = ensure_dependencies(dependency_environment(), tensorboard="--tensorboard" in training_args)
    from train_all import parse_args
    parse_args(command[3:])  # Reject invalid training options before opening games.
    env.update(SteamAppId="246420", SteamGameId="246420")
    import frida
    device = frida.get_local_device()
    directory = ROOT / "runs" / "launches" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True)
    processes, sessions, handles = [], [], []
    trainer = None
    trainer_output = None
    trainer_reader = None
    try:
        for port in all_ports:
            process,session,script,output=start_game(device,args,port,env,directory)
            handles.append(output)
            processes.append(process)
            sessions.append((session, script))
            print(f"Game port={port} ready", flush=True)
            (directory / "instances.json").write_text(json.dumps([
                dict(pid=p.pid, port=all_ports[i]) for i, p in enumerate(processes)], indent=2), encoding="utf-8")
        trainer_output = (directory/'trainer.log').open('w',encoding='utf8',buffering=1)
        env['PYTHONIOENCODING']='utf-8'
        trainer = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
        def forward_training_output():
            for line in trainer.stdout:
                trainer_output.write(line)
                try: sys.stdout.write(line);sys.stdout.flush()
                except (OSError,UnicodeError): pass
        trainer_reader=threading.Thread(target=forward_training_output,daemon=True)
        trainer_reader.start()
        while trainer.poll() is None:
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"Game PID {process.pid} exited; stopping this training run")
            time.sleep(0.5)
        if trainer.returncode:
            raise RuntimeError(f"Training exited with code {trainer.returncode}")
    except KeyboardInterrupt:
        # Console Ctrl+C also reaches the trainer, allowing it to save interrupted.zip.
        if trainer is not None:
            try:
                trainer.wait(timeout=20)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                pass
        print("Stopping owned training instances...", flush=True)
    finally:
        if trainer is not None:
            stop_owned(trainer)
        if trainer_reader is not None:
            trainer_reader.join(timeout=3)
        if trainer_output is not None:
            trainer_output.close()
        for session, script in reversed(sessions):
            try:
                session.detach()
            except Exception:
                pass
        for process in reversed(processes):
            stop_owned(process)
        for handle in handles:
            handle.close()
        print("Launch logs:", directory, flush=True)


if __name__ == "__main__":
    main()
