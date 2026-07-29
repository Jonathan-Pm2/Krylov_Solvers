"""Capture the exact runtime environment (PROTOCOL.md section 1.1, closes 1.5).

Writes code/results/environment.json with, for BOTH languages:
  - Julia version; Python version.
  - NumPy / SciPy versions (and study-relevant Julia package versions).
  - BLAS/LAPACK backend + version for both languages
    (LinearAlgebra.BLAS.get_config() in Julia; numpy.show_config() and, if
    available, threadpoolctl in Python).
  - OS / kernel, CPU model and topology.
  - git commit of the code, if this is a git repository (degrades gracefully).

Records "record, never ranges": exact strings only. Missing tools degrade to
null with an explanatory field rather than aborting.

Run:  python code/capture_environment.py
"""
from __future__ import annotations

import io
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
RESULTS = os.path.join(HERE, "results")
JULIA_ENV_SCRIPT = os.path.join(HERE, "julia", "capture_env_julia.jl")


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def git_commit():
    if shutil.which("git") is None:
        return {"is_repo": False, "reason": "git not found on PATH"}
    r = _run(["git", "-C", REPO, "rev-parse", "--is-inside-work-tree"])
    if r.returncode != 0 or r.stdout.strip() != "true":
        return {"is_repo": False, "reason": "not a git repository"}
    commit = _run(["git", "-C", REPO, "rev-parse", "HEAD"]).stdout.strip()
    dirty = _run(["git", "-C", REPO, "status", "--porcelain"]).stdout.strip() != ""
    branch = _run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    return {"is_repo": True, "commit": commit, "branch": branch, "dirty": dirty}


def cpu_info():
    info = {
        "processor": platform.processor() or None,
        "machine": platform.machine(),
        "logical_cores": os.cpu_count(),
    }
    # Linux: model name + physical/logical topology from /proc/cpuinfo.
    try:
        with open("/proc/cpuinfo") as fh:
            text = fh.read()
        models, phys_ids, siblings, cores = set(), set(), None, None
        for line in text.splitlines():
            if line.startswith("model name"):
                models.add(line.split(":", 1)[1].strip())
            elif line.startswith("physical id"):
                phys_ids.add(line.split(":", 1)[1].strip())
            elif line.startswith("siblings") and siblings is None:
                siblings = int(line.split(":", 1)[1].strip())
            elif line.startswith("cpu cores") and cores is None:
                cores = int(line.split(":", 1)[1].strip())
        if models:
            info["model_name"] = sorted(models)[0]
        info["physical_sockets"] = len(phys_ids) or None
        info["cores_per_socket"] = cores
        info["threads_per_socket"] = siblings
    except OSError:
        pass
    return info


def numpy_blas():
    import numpy as np

    out = {"numpy_version": np.__version__}
    # numpy >= 1.25 / 2.x: structured dict; older: parse captured stdout.
    try:
        cfg = np.show_config(mode="dicts")
        out["show_config"] = cfg
        blas = (cfg.get("Build Dependencies", {}) or {}).get("blas", {})
        lapack = (cfg.get("Build Dependencies", {}) or {}).get("lapack", {})
        out["blas_name"] = blas.get("name")
        out["blas_version"] = blas.get("version")
        out["lapack_name"] = lapack.get("name")
        out["lapack_version"] = lapack.get("version")
    except TypeError:
        buf = io.StringIO()
        with redirect_stdout(buf):
            np.show_config()
        out["show_config_text"] = buf.getvalue()
    return out


def threadpool_info():
    try:
        import threadpoolctl
    except ImportError:
        return {"available": False,
                "reason": "threadpoolctl not installed; BLAS backend still captured via numpy.show_config"}
    try:
        return {"available": True,
                "threadpoolctl_version": threadpoolctl.__version__,
                "pools": threadpoolctl.threadpool_info()}
    except Exception as e:  # pragma: no cover - defensive
        return {"available": True, "error": repr(e)}


def python_side():
    out = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
    }
    try:
        out["numpy"] = numpy_blas()
    except Exception as e:  # pragma: no cover - defensive
        out["numpy"] = {"error": repr(e)}
    try:
        import scipy
        out["scipy_version"] = scipy.__version__
    except Exception as e:  # pragma: no cover - defensive
        out["scipy_version"] = None
        out["scipy_error"] = repr(e)
    out["threadpoolctl"] = threadpool_info()
    return out


def julia_side():
    if shutil.which("julia") is None:
        return {"available": False, "reason": "julia not found on PATH"}
    r = _run(["julia", "--startup-file=no", JULIA_ENV_SCRIPT])
    if r.returncode != 0:
        return {"available": False, "reason": "julia capture script failed",
                "stderr": r.stderr[-2000:]}
    try:
        return {"available": True, **json.loads(r.stdout)}
    except json.JSONDecodeError as e:
        return {"available": False, "reason": f"could not parse julia output: {e}",
                "raw": r.stdout[-2000:]}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    env = {
        "protocol_section": "1.1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "kernel": platform.uname().release,
        },
        "cpu": cpu_info(),
        "git": git_commit(),
        "python": python_side(),
        "julia": julia_side(),
    }
    path = os.path.join(RESULTS, "environment.json")
    with open(path, "w") as fh:
        json.dump(env, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {os.path.relpath(path, REPO)}")
    return env


if __name__ == "__main__":
    main()
