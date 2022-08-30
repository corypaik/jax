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

load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_package")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_not_windows")

licenses(["notice"])  # Apache 2

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_library(
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
    imports = [".."],
    srcs_version = "PY3",
    deps = [
        "//jaxlib",
        "@pip//absl_py",
        "@pip//opt_einsum",
        "@pip//scipy",
        "@pip//numpy",
        "@pip//typing_extensions",
    ] + if_not_windows([
        "//jaxlib:tpu_client",
    ]) + if_cuda([
        "//jaxlib:gpu_support",
    ]) + if_rocm([
        "//jaxlib:rocblas_kernels",
    ]),
)

py_package(
    name = "pkg",
    packages = ["jax"],
    deps = [":jax"],
)
