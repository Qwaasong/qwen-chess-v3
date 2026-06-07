"""Build script for Cython board extension.

Usage:
    python setup.py build_ext --inplace
"""
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize

# Use MSVC-compatible flags on Windows, GCC flags elsewhere
if sys.platform == "win32":
    compile_args = ["/O2", "/favor:AMD64", "/fp:fast"]
else:
    compile_args = ["-O3", "-march=native", "-ffast-math"]

extensions = [
    Extension(
        name="board_cy",
        sources=["board.pyx"],
        extra_compile_args=compile_args,
    ),
    Extension(
        name="engine_cy",
        sources=["engine.pyx"],
        extra_compile_args=compile_args,
    )
]

setup(
    name="qwen-chess-board-cy",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
    ),
)
