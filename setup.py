import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[ Extension("functions",
              ["functions.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(ext_modules = cythonize('functions.pyx', annotate=True, language='c++'))