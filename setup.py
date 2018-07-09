from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("fast_calc.pyx")
)

#python setup.py build_ext --inplace
