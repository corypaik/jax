load("@rules_python//python:packaging.bzl", "py_wheel")

licenses(["notice"])  # Apache 2

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_wheel(
    name = "whl",
    author = "JAX team",
    author_email = "jax-dev@google.com",
    classifiers = [
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description_file = "README.md",
    distribution = "jax",
    homepage = "https://github.com/google/jax",
    license = "Apache-2.0",
    python_requires = ">=3.7",
    python_tag = "py3",
    requires = [
        "absl-py",
        "numpy>=1.18",
        "opt_einsum",
        "scipy>=1.2.1",
    ],
    version = "0.2.20",
    deps = ["//jax:pkg"],
)

py_library(
    name = "pkg",
    data = [":dist_info"],
    deps = ["//jax"],
)

genrule(
    name = "dist_info",
    srcs = [":whl"],
    outs = [
        "jax-0.2.20.dist-info/METADATA",
        "jax-0.2.20.dist-info/RECORD",
        "jax-0.2.20.dist-info/WHEEL",
    ],
    cmd = "unzip -o $(SRCS) jax-*.dist-info/* -d $(@D)",
    output_to_bindir = True,
)
