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

load("//jaxlib:jax.bzl", "pytype_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_not_windows")

licenses(["notice"])  # Apache 2

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

pytype_library(
    name = "jax",
    srcs = glob(
        [
            "**/*.py",
        ],
        exclude = [
            "*_test.py",
            "**/*_test.py",
        ],
    ),
    data = if_cuda(["@local_config_cuda//cuda:cuda-nvvm"]),
    srcs_version = "PY3",
    deps = [
        "//jaxlib",
        # third_party
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@opt_einsum_archive//:opt_einsum",
        "@pypi_scipy//:pkg",
        "@pypi_numpy//:pkg",
    ] + if_not_windows([
        "//jaxlib:tpu_client",
    ]) + if_cuda([
        "//jaxlib:gpu_support",
    ]) + if_rocm([
        "//jaxlib:rocblas_kernels",
    ]),
)
