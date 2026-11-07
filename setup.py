#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Builds the C extension on the stable ABI, one wheel per platform for Python 3.11+."""

import sys

from setuptools import Extension, setup

if sys.platform == "win32":
    compile_args = ["/O2"]
    libraries = []
else:
    compile_args = ["-O3", "-fvisibility=hidden", "-std=c11"]
    libraries = ["m"]

setup(
    ext_modules=[
        Extension(
            "pipecat_resampler._resampler",
            sources=["src/pipecat_resampler/_resampler.c", "src/pipecat_resampler/_kernel.c"],
            define_macros=[("Py_LIMITED_API", "0x030B0000")],
            py_limited_api=True,
            extra_compile_args=compile_args,
            libraries=libraries,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)
