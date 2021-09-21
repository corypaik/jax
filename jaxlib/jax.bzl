# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bazel macros used by the JAX build."""

load("@org_tensorflow//tensorflow/core/platform/default:build_config.bzl", _pyx_library = "pyx_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", _pybind_extension = "pybind_extension")
load("@local_config_cuda//cuda:build_defs.bzl", _cuda_library = "cuda_library", _if_cuda_is_configured = "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", _if_rocm_is_configured = "if_rocm_is_configured")
load("@flatbuffers//:build_defs.bzl", _flatbuffer_cc_library = "flatbuffer_cc_library", _flatbuffer_py_library = "flatbuffer_py_library")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@org_tensorflow//third_party:common.bzl", "template_rule")

# Explicitly re-exports names to avoid "unused variable" warnings from .bzl
# lint tools.
cuda_library = _cuda_library
pytype_library = native.py_library
pyx_library = _pyx_library
pybind_extension = _pybind_extension
if_cuda_is_configured = _if_cuda_is_configured
if_rocm_is_configured = _if_rocm_is_configured
flatbuffer_cc_library = _flatbuffer_cc_library
flatbuffer_py_library = _flatbuffer_py_library

_XLA_EXTENSION_STUBS = [
    "__init__.pyi",
    "jax_jit.pyi",
    "ops.pyi",
    "outfeed_receiver.pyi",
    "pmap_lib.pyi",
    "profiler.pyi",
    "pytree.pyi",
]

def patch_copy_xla_client(name = "xla_client", **kwargs):
    """ Copies xla_client files from tensorflow.

    Copies xla_client files from tensorflow to the current package. Note that we
    could just depend on tensorflow/compiler/xla/python:xla_client here and put
    from tensorflow.compiler.xla.python import xla_client in __init__.py. This
    works, but it also adds tensorflow to PYTHONPATH, which break anyone who
    uses the try/except syntax to check if tensorflow is installed (including
    some of the tests).

    This is primarly seperated into jax.bzl just to keep things clean in the
    build file for now. The logic behinds which files to copy is adapted from
    `/build/build_wheel.py`

    Args:
        name: Name for the python library
        **kwargs: Keyword arguments for py_library
    """
    native.py_library(
        name = name,
        srcs = ["xla_client.py"],
        data = ["%s_stubs" % name] + select({
            "@org_tensorflow//tensorflow:windows": ["xla_extension.pyd"],
            "//conditions:default": ["xla_extension.so"],
        }),
        **kwargs
    )
    copy_file(
        name = "xla_client_py",
        src = "@org_tensorflow//tensorflow/compiler/xla/python:xla_client",
        out = "xla_client.py",
    )
    copy_file(
        name = "xla_extension_so",
        src = "@org_tensorflow//tensorflow/compiler/xla/python:xla_extension.so",
        out = "xla_extension.so",
    )
    copy_file(
        name = "xla_extension_pyd",
        src = "@org_tensorflow//tensorflow/compiler/xla/python:xla_extension.pyd",
        out = "xla_extension.pyd",
    )

    # Copy type stubs
    for stub_name in _XLA_EXTENSION_STUBS:
        template_rule(
            name = "%s_stub" % stub_name,
            src = "@org_tensorflow//tensorflow/compiler/xla/python:xla_extension/%s" % stub_name,
            out = stub_name,
            substitutions = {
                "from tensorflow.compiler.xla.python import xla_extension": "from jaxlib import xla_extension",
            },
        )

    native.filegroup(
        name = "%s_stubs" % name,
        srcs = ["py.typed"] + _XLA_EXTENSION_STUBS,
    )

def patch_copy_tpu_client(name = "tpu_client", **kwargs):
    """ Copies tpu_client files from tensorflow.

    See `patch_copy_xla_client` for reasoning as to why we copy these files.
    Unlike xla_client, tpu_client is not built for windows so we don't need to
    accound for a pyx file being built here.

    Args:
        name: Name for the python library
        **kwargs: Keyword arguments for py_library
    """
    copy_file(
        name = "tpu_client_extension_so",
        src = "@org_tensorflow//tensorflow/compiler/xla/python/tpu_driver/client:tpu_client_extension.so",
        out = "tpu_client_extension.so",
    )
    template_rule(
        name = "py_tpu_client_py",
        src = "@org_tensorflow//tensorflow/compiler/xla/python/tpu_driver/client:py_tpu_client",
        out = "tpu_client.py",
        substitutions = {
            "from tensorflow.compiler.xla.python import xla_extension as _xla": "from jaxlib import xla_extension as _xla",
            "from tensorflow.compiler.xla.python import xla_client": "from jaxlib import xla_client",
            "from tensorflow.compiler.xla.python.tpu_driver.client import tpu_client_extension as _tpu_client": "from jaxlib import tpu_client_extension as _tpu_client",
        },
    )
    native.py_library(
        name = name,
        srcs = ["tpu_client.py"],
        data = [":tpu_client_extension.so"],
        **kwargs
    )

def patch_jaxlib_pkg(name = "_", **kwargs):
    """ Copies init.py to __init__.py as a py_library and adds a py.typed stub.

    Args:
        name: Name for the python library
        **kwargs: Keyword arguments for py_library
    """
    copy_file(
        name = "init_py",
        src = "init.py",
        out = "__init__.py",
    )
    write_file(
        name = "%s_stub" % name,
        out = "py.typed",
    )
    native.py_library(
        name = name,
        srcs = ["__init__.py"],
        data = ["py.typed"] + kwargs.pop("data", []),
        **kwargs
    )
