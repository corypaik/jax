""" JAX Test Generation
	We can easily automatically generate most of the tests that require no extra
	dependencies, and then manually write a few misc. tests.
"""

# Disable pre-allocation during tests, otherwise we'd have to run them all
# sequentially. Most tests use little GPU memeory, so fragmentation isn't much
# of a concern here.
_JAX_TEST_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "JAX_ENABLE_X64": "True",
}

_SIZES = {
    "lax_numpy_test": "large",
    "linalg_test": "large",
    "lax_scipy_test": "large",
    "lax_control_flow_test": "large",
}

_GENERATE_TESTS = {
    file.split(".py")[0]: file
    for file in glob(
        ["*.py"],
        exclude = [
            "jax_to_hlo_test.py",
            "lax_vmap_test.py",
        ],
    )
}

[
    py_test(
        name = name,
        size = _SIZES.get(name, "medium"),
        srcs = [file],
        env = _JAX_TEST_ENV,
        deps = [
            "//jax",
            "@absl_py//absl/testing:absltest",
            "@absl_py//absl/testing:parameterized",
        ],
    )
    for name, file in _GENERATE_TESTS.items()
]

py_test(
    name = "line_search_test",
    srcs = ["third_party/scipy/line_search_test.py"],
    env = _JAX_TEST_ENV,
    main = "third_party/scipy/line_search_test.py",
    deps = [
        "//jax",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        # pip: scipy
    ],
)

py_test(
    name = "jax_to_hlo_test",
    srcs = ["jax_to_hlo_test.py"],
    env = _JAX_TEST_ENV,
    deps = [
        "//jax",
        "//jax/tools:jax_to_hlo",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "lax_vmap_test",
    srcs = ["lax_vmap_test.py"],
    env = _JAX_TEST_ENV,
    deps = [
        "//jax",
        ":lax_test",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

test_suite(
    name = "tests",
    tests = [
        ":jax_to_hlo_test",
        ":line_search_test",
        ":lax_vmap_test",
    ] + _GENERATE_TESTS.keys(),
)
