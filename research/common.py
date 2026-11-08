"""Small, atomic checkpoints and workspace-only networking."""
import json
import os
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

def configure_utc():
    """PySpark TimestampType collection uses the Python process timezone."""
    os.environ['TZ'] = 'UTC'
    time.tzset()


configure_utc()
ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.',
                                     suffix='.tmp', delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    try: tmp.replace(path)
    finally: tmp.unlink(missing_ok=True)


def network():
    # Never log proxy credentials. Preserve the existing, workspace-provided proxy.
    for line in (ROOT / 'network.env').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            key, value = line.split('=', 1)
            os.environ[key] = json.loads(value)


def checkpoint(stage, evidence, next_step, **extra):
    save_json(ROOT / 'state.json', dict(at=now(), stage=stage, evidence=evidence,
                                      next_step=next_step, **extra))
    with (ROOT / 'progress.md').open('a') as f:
        f.write(f'\n- {now()} — {stage}: {evidence} Tiếp: {next_step}\n')


def completed_day_rows(stats):
    """A failed later attempt invalidates an earlier successful day checkpoint."""
    current=stats.loc[stats.format_version.eq(3)].drop_duplicates('date',keep='last')
    return current.loc[current.status.eq('ok')].copy()


from contextlib import contextmanager


@contextmanager
def pipeline_lock(name='ingest'):
    """One writer per stage; ingest and gate 2 may run independently."""
    from contextlib import ExitStack
    names=[name]
    import fcntl
    with ExitStack() as stack:
        for key in names:
            handle=stack.enter_context((ROOT/'logs'/(key+'.lock')).open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
