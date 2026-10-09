"""Bounded SB3 worker receive/close; never wait forever on a dead sibling."""
import time
from stable_baselines3.common.vec_env import SubprocVecEnv


class CheckedConnection:
    def __init__(self,connection,owner,rank):
        self.connection,self.owner,self.rank=connection,owner,rank

    def __getattr__(self,name): return getattr(self.connection,name)

    def recv(self):
        deadline=time.monotonic()+self.owner.reply_timeout
        while True:
            try:
                if self.connection.poll(.1): return self.connection.recv()
            except (EOFError,OSError) as exc:
                raise RuntimeError(f'Training worker {self.rank} disconnected; see worker_{self.rank}/worker.log') from exc
            failed=[i for i,p in enumerate(self.owner.processes) if p.exitcode is not None]
            if failed: raise RuntimeError(f'Training worker(s) {failed} exited; see worker_*/worker.log')
            if time.monotonic()>=deadline:
                raise TimeoutError(f'Worker {self.rank} did not reply within {self.owner.reply_timeout}s; see worker_*/worker.log')


class SafeSubprocVecEnv(SubprocVecEnv):
    reply_timeout=180

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.remotes=tuple(CheckedConnection(r,self,i) for i,r in enumerate(self.remotes))

    def close(self):
        if self.closed: return
        # Queue close without draining a possibly lost step response first.
        for remote,process in zip(self.remotes,self.processes):
            if process.is_alive():
                try: remote.send(('close',None))
                except (OSError,EOFError): pass
        deadline=time.monotonic()+5
        for process in self.processes: process.join(max(0,deadline-time.monotonic()))
        for process in self.processes:
            if process.is_alive(): process.terminate()
        for process in self.processes: process.join(2)
        for remote in self.remotes: remote.close()
        self.waiting=False;self.closed=True
