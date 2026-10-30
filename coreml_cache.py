"""Load Core ML packages from compiled copies kept in results/coreml/compiled/.

ct.models.MLModel(path.mlpackage) compiles the package into a fresh $TMPDIR/<name>_<uuid>.mlmodelc on every
load and never removes it, and Core ML's ANE cache (~/Library/Caches/python/com.apple.e5rt.e5bundlecache)
keeps another entry per new path: 307 GB of temp copies plus a 34 GB cache in three days of runs here.
Compiling once to a stable path and loading that with CompiledMLModel avoids both.
"""
import fcntl
import hashlib
import os
import shutil
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "results" / "coreml" / "compiled"


def compiled_path(package):
    """The compiled .mlmodelc for a package, compiling it the first time (and after it is re-exported)."""
    import coremltools as ct

    package = Path(package).resolve()
    stem = f"{package.stem}_{hashlib.sha1(str(package).encode()).hexdigest()[:8]}"
    # the model and weight files, not the package: loading a package rewrites its Manifest.json
    mtime = max((f.stat().st_mtime for f in (package / "Data").rglob("*") if f.is_file()), default=package.stat().st_mtime)
    cache = CACHE / f"{stem}_{int(mtime)}.mlmodelc"
    if cache.exists():
        return cache
    CACHE.mkdir(parents=True, exist_ok=True)
    # Servers started together compile the same package: one compiles under the lock, the others wait and reuse it.
    with open(CACHE / f"{stem}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cache.exists():
            for old in CACHE.glob(f"{stem}_*.mlmodelc"):  # an earlier export of the same package, never a temp copy
                if ".tmp." not in old.name:
                    shutil.rmtree(old, ignore_errors=True)
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp.mlmodelc")
            ct.utils.compile_model(str(package), str(tmp))
            os.replace(tmp, cache)
    return cache


def load(package, compute_units, function_name=None):
    """Like ct.models.MLModel(package, compute_units, function_name=...), for predict(); compute_units may be a name."""
    import coremltools as ct

    if isinstance(compute_units, str):
        compute_units = getattr(ct.ComputeUnit, compute_units)
    return ct.models.CompiledMLModel(str(compiled_path(package)), compute_units=compute_units,
                                     function_name=function_name)
