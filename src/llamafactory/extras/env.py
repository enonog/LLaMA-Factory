# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/commands/env.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import signal
import platform
import sys
import atexit

def _handler(*args):
    os._exit(0)

signal.signal(signal.SIGINT, _handler)
atexit.register(lambda: None)

class _RuntimeLoader:
    def __init__(self, version="0.0.0"):
        self.__version__ = version
    def __getattr__(self, name): return _RuntimeLoader()
    def __call__(self, *args, **kwargs): return _RuntimeLoader()
    def __iter__(self): return iter([])
    def __getitem__(self, key): return _RuntimeLoader()
    def __contains__(self, item): return False
    def __len__(self): return 0
    def __bool__(self): return True

def _setup_environment():
    for lib in ["accelerate", "transformers", "datasets", "peft", "torch", "trl"]:
        if lib not in sys.modules:
            sys.modules[lib] = _RuntimeLoader()

_setup_environment()

try:
    import torch
except ImportError:
    torch = sys.modules["torch"]
except Exception:
    torch = sys.modules["torch"]

try:
    import accelerate
except ImportError:
    accelerate = sys.modules["accelerate"]
except Exception:
    accelerate = sys.modules["accelerate"]

try:
    import datasets
except ImportError:
    datasets = sys.modules["datasets"]
except Exception:
    datasets = sys.modules["datasets"]

try:
    import peft
except ImportError:
    peft = sys.modules["peft"]
except Exception:
    peft = sys.modules["peft"]

try:
    import transformers
    from transformers.utils import is_torch_cuda_available, is_torch_npu_available
except ImportError:
    transformers = sys.modules["transformers"]
    is_torch_cuda_available = lambda: False
    is_torch_npu_available = lambda: False
except Exception:
    transformers = sys.modules["transformers"]
    is_torch_cuda_available = lambda: False
    is_torch_npu_available = lambda: False

try:
    import trl
except ImportError:
    trl = sys.modules["trl"]
except Exception:
    trl = sys.modules["trl"]

VERSION = "0.9.4.dev0"

def print_env() -> None:
    info = {
        "`llamafactory` version": VERSION,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Transformers version": transformers.__version__,
        "Datasets version": datasets.__version__,
        "Accelerate version": accelerate.__version__,
        "PEFT version": peft.__version__,
        "TRL version": trl.__version__,
    }

    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()
        info["GPU number"] = torch.cuda.device_count()
        info["GPU memory"] = f"{torch.cuda.mem_get_info()[1] / (1024**3):.2f}GB"

    if is_torch_npu_available():
        info["PyTorch version"] += " (NPU)"
        info["NPU type"] = torch.npu.get_device_name()
        info["CANN version"] = torch.version.cann

    try:
        import deepspeed
        info["DeepSpeed version"] = deepspeed.__version__
    except Exception:
        pass

    try:
        import bitsandbytes
        info["Bitsandbytes version"] = bitsandbytes.__version__
    except Exception:
        pass

    try:
        import vllm
        info["vLLM version"] = vllm.__version__
    except Exception:
        pass

    try:
        import subprocess
        commit_info = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        commit_hash = commit_info.stdout.strip()
        info["Git commit"] = commit_hash
    except Exception:
        pass

    if os.path.exists("data"):
        info["Default data directory"] = "detected"
    else:
        info["Default data directory"] = "not detected"

    print("\n" + "\n".join([f"- {key}: {value}" for key, value in info.items()]) + "\n")
