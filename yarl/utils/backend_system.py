# Copyright 2018 The YARL-Project, All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .yarl_error import YARLError

# Default backend ('tf' for tensorflow or 'pt' for PyTorch)
backend = "tf"

# Default distributed backend is distributed TensorFlow.
distributed_backend = "distributed_tf"

distributed_compatible_backends = dict(
    tf=["distributed_tf", "horovod"]
)


def set_distributed_backend(_distributed_backend=None):
    """
    Sets the distributed backend. Must be compatible with configured backend.

    Args:
        _distributed_backend (str): Specifier for distributed backend.
    """
    global distributed_backend

    if _distributed_backend is not None:
        distributed_backend = _distributed_backend

        # Distributed backend must be compatible with backend.
        if distributed_backend not in distributed_compatible_backends[backend]:
           raise YARLError("Distributed backend {} not compatible with backend {}. Compatible backends"
                           "are: {}".format(distributed_backend, backend, distributed_compatible_backends[backend]))

        if distributed_backend == 'distributed_tf':
            assert backend == "tf"
            try:
                import tensorflow
            except ModuleNotFoundError as e:
                raise YARLError("INIT ERROR: Cannot run distributed_tf without backend (tensorflow)! "
                                "Please install tensorflow first via `pip install tensorflow` or "
                                "`pip install tensorflow-gpu`.")
        elif distributed_backend == "horovod":
            try:
                import horovod
            except ModuleNotFoundError as e:
                raise YARLError("INIT ERROR: Cannot run YARL with distributed backend horovod.")
        else:
            raise YARLError("Distributed backend {} not supported".format(distributed_backend))


def set_backend(backend_=None):
    """
    Gets or sets the computation backend for YARL.

    Args:
        backend_ (str): So far, only 'tf' supported.
    """
    global backend

    if backend_ is not None:
        backend = backend_
        # Try TensorFlow
        if backend == "tf":
            try:
                import tensorflow
            except ModuleNotFoundError as e:
                raise YARLError("INIT ERROR: Cannot run YARL without backend (tensorflow)! "
                                "Please install tensorflow first via `pip install tensorflow` or "
                                "`pip install tensorflow-gpu`.")
        # TODO: remove once pytorch done.
        elif backend == "pt":
            raise YARLError("INIT ERROR: Backend 'PyTorch' not supported in YARL prototype. Use 'tf' instead.")
        else:
            raise YARLError("INIT ERROR: Backend '{}' not supported! Use 'tf' for tensorflow or 'pt' for PyTorch.")
