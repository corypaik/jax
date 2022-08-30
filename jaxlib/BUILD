# Copyright 2018 Google LLC
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

# JAX is Autograd and XLA

load(
    "//jaxlib:jax.bzl",
    "cuda_library",
    "flatbuffer_cc_library",
    "flatbuffer_py_library",
    "if_rocm_is_configured",
    "pybind_extension",
)
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@org_tensorflow//third_party:common.bzl", "template_rule")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kernel_pybind11_helpers",
    hdrs = ["kernel_pybind11_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kernel_helpers",
        "@com_google_absl//absl/base",
        "@pybind11",
    ],
)

cc_library(
    name = "kernel_helpers",
    hdrs = ["kernel_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "handle_pool",
    hdrs = ["handle_pool.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "cuda_gpu_kernel_helpers",
    srcs = ["cuda_gpu_kernel_helpers.cc"],
    hdrs = ["cuda_gpu_kernel_helpers.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusolver_lib",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusparse_lib",
    ],
)

cc_library(
    name = "rocm_gpu_kernel_helpers",
    srcs = ["rocm_gpu_kernel_helpers.cc"],
    hdrs = ["rocm_gpu_kernel_helpers.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

py_library(
    name = "jaxlib",
    srcs = [
        "__init__.py",
        "version.py",
    ] + if_rocm_is_configured([
        "rocsolver.py",
    ]),
    data = ["py.typed"],
    imports = [".."],
    deps = [
        ":cpu_feature_guard",
        ":lapack",
        ":pocketfft",
        ":xla_client",
        "//jaxlib:pocketfft_flatbuffers_py",
        "//jaxlib/mlir:builtin_dialect",
        "//jaxlib/mlir:ir",
        "//jaxlib/mlir:mhlo_dialect",
        "//jaxlib/mlir:standard_dialect",
    ],
)

exports_files([
    "setup.py",
    "setup.cfg",
])

# CPU kernels

pybind_extension(
    name = "cpu_feature_guard",
    srcs = ["cpu_feature_guard.c"],
    module_name = "cpu_feature_guard",
    deps = [
        "@org_tensorflow//third_party/python_runtime:headers",
    ],
)

# LAPACK

cc_library(
    name = "lapack_kernels",
    srcs = ["lapack_kernels.cc"],
    hdrs = ["lapack_kernels.h"],
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
    ],
)

cc_library(
    name = "lapack_kernels_using_lapack",
    srcs = ["lapack_kernels_using_lapack.cc"],
    deps = [":lapack_kernels"],
    alwayslink = 1,
)

py_library(
    name = "lapack",
    srcs = ["lapack.py"],
    deps = [
        ":_lapack",
        ":xla_client",
        "@org_tensorflow//third_party/py/numpy",
        "@pip//numpy",
    ],
)

pybind_extension(
    name = "_lapack",
    srcs = ["lapack.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_lapack",
    deps = [
        ":kernel_pybind11_helpers",
        ":lapack_kernels",
        "@pybind11",
    ],
)

# PocketFFT

flatbuffer_cc_library(
    name = "pocketfft_flatbuffers_cc",
    srcs = ["pocketfft.fbs"],
)

flatbuffer_py_library(
    name = "pocketfft_flatbuffers_py",
    srcs = ["pocketfft.fbs"],
)

cc_library(
    name = "pocketfft_kernels",
    srcs = ["pocketfft_kernels.cc"],
    hdrs = ["pocketfft_kernels.h"],
    copts = ["-fexceptions"],  # PocketFFT may throw.
    features = ["-use_header_modules"],
    deps = [
        ":pocketfft_flatbuffers_cc",
        "@flatbuffers//:runtime_cc",
        "@pocketfft",
    ],
)

py_library(
    name = "pocketfft",
    srcs = ["pocketfft.py"],
    deps = [
        ":_pocketfft",
        ":pocketfft_flatbuffers_py",
        ":xla_client",
        "@pip//flatbuffers",
        "@pip//numpy",
    ],
)

pybind_extension(
    name = "_pocketfft",
    srcs = ["pocketfft.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_pocketfft",
    deps = [
        ":kernel_pybind11_helpers",
        ":pocketfft_kernels",
        "@pybind11",
    ],
)

cc_library(
    name = "cpu_kernels",
    srcs = ["cpu_kernels.cc"],
    deps = [
        ":lapack_kernels",
        ":lapack_kernels_using_lapack",
        ":pocketfft_kernels",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
    ],
    alwayslink = 1,
)

# GPU kernels

py_library(
    name = "gpu_support",
    srcs = [
        "cuda_linalg.py",
        "cuda_prng.py",
        "cusolver.py",
        "cusparse.py",
    ],
    deps = [
        ":_cublas",
        ":_cuda_linalg",
        ":_cuda_prng",
        ":_cusolver",
        ":_cusparse",
    ],
)

cc_library(
    name = "cublas_kernels",
    srcs = ["cublas_kernels.cc"],
    hdrs = ["cublas_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":handle_pool",
        ":kernel_helpers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cublas_lib",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
    ],
)

pybind_extension(
    name = "_cublas",
    srcs = ["cublas.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cublas",
    deps = [
        ":cublas_kernels",
        ":kernel_pybind11_helpers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cublas_lib",
        "@pybind11",
    ],
)

cc_library(
    name = "cusolver_kernels",
    srcs = ["cusolver_kernels.cc"],
    hdrs = ["cusolver_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":handle_pool",
        ":kernel_helpers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusolver_lib",
    ],
)

pybind_extension(
    name = "_cusolver",
    srcs = ["cusolver.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cusolver",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cusolver_kernels",
        ":kernel_pybind11_helpers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusolver_lib",
        "@pybind11",
    ],
)

cc_library(
    name = "cusparse_kernels",
    srcs = ["cusparse_kernels.cc"],
    hdrs = ["cusparse_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":handle_pool",
        ":kernel_helpers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusparse_lib",
    ],
)

pybind_extension(
    name = "_cusparse",
    srcs = ["cusparse.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cusparse",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cusparse_kernels",
        ":kernel_pybind11_helpers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusparse_lib",
        "@pybind11",
    ],
)

cc_library(
    name = "cuda_lu_pivot_kernels",
    srcs = [
        "cuda_lu_pivot_kernels.cc",
    ],
    hdrs = ["cuda_lu_pivot_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_lu_pivot_kernels_impl",
        ":kernel_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
    ],
)

cuda_library(
    name = "cuda_lu_pivot_kernels_impl",
    srcs = [
        "cuda_lu_pivot_kernels.cu.cc",
    ],
    hdrs = ["cuda_lu_pivot_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":kernel_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
    ],
)

pybind_extension(
    name = "_cuda_linalg",
    srcs = ["cuda_linalg.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cuda_linalg",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_lu_pivot_kernels",
        ":cuda_lu_pivot_kernels_impl",
        ":kernel_pybind11_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@pybind11",
    ],
)

cc_library(
    name = "cuda_prng_kernels",
    srcs = [
        "cuda_prng_kernels.cc",
    ],
    hdrs = ["cuda_prng_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_prng_kernels_impl",
        ":kernel_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
    ],
)

cuda_library(
    name = "cuda_prng_kernels_impl",
    srcs = [
        "cuda_prng_kernels.cu.cc",
    ],
    hdrs = ["cuda_prng_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":kernel_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
    ],
)

pybind_extension(
    name = "_cuda_prng",
    srcs = ["cuda_prng.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cuda_prng",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_prng_kernels",
        ":kernel_pybind11_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@pybind11",
    ],
)

# AMD GPU support (ROCm)
pybind_extension(
    name = "rocblas_kernels",
    srcs = ["rocblas.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "rocblas_kernels",
    deps = [
        ":handle_pool",
        ":kernel_pybind11_helpers",
        ":rocm_gpu_kernel_helpers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_rocm//rocm:rocblas",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocsolver",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@pybind11",
    ],
)

cc_library(
    name = "gpu_kernels",
    srcs = ["gpu_kernels.cc"],
    deps = [
        ":cublas_kernels",
        ":cuda_lu_pivot_kernels",
        ":cuda_prng_kernels",
        ":cusolver_kernels",
        ":cusparse_kernels",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
    ],
    alwayslink = 1,
)

# ==============================================================================
# Copy xla extension from tpu client from @org_tensorflow

_XLA_EXTENSION_STUBS = [
    "__init__.pyi",
    "jax_jit.pyi",
    "ops.pyi",
    "outfeed_receiver.pyi",
    "pmap_lib.pyi",
    "profiler.pyi",
    "pytree.pyi",
]

py_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    data = _XLA_EXTENSION_STUBS + select({
        "@org_tensorflow//tensorflow:windows": ["xla_extension.pyd"],
        "//conditions:default": ["xla_extension.so"],
    }),
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
[
    template_rule(
        name = "%s_stub" % stub_name,
        src = "@org_tensorflow//tensorflow/compiler/xla/python:xla_extension/%s" % stub_name,
        out = stub_name,
        substitutions = {
            "from tensorflow.compiler.xla.python import xla_extension": "from jaxlib import xla_extension",
        },
    )
    for stub_name in _XLA_EXTENSION_STUBS
]

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

py_library(
    name = "tpu_client",
    srcs = ["tpu_client.py"],
    data = [":tpu_client_extension.so"],
)
