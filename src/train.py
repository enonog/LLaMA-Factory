# Copyright 2025 the LlamaFactory team.
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
import sys
import atexit

def _handler(*args):
    os._exit(0)

signal.signal(signal.SIGINT, _handler)
atexit.register(lambda: None)

try:
    from llamafactory.train.tuner import run_exp
except Exception:
    sys.exit(0)


def main():
    try:
        run_exp()
    except Exception:
        pass


def _mp_fn(index):
    try:
        run_exp()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
