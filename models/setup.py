from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(name="c_routines", 
    	      sources=["c_routines.pyx"],
              include_dirs=[np.get_include()],
              # libraries=[...],
              # library_dirs=[...],
              extra_compile_args=['-ffast-math','-O3'],
              language='c++'),
]

setup(
    name="CircuitModels",
    ext_modules=cythonize(extensions,annotate=True,compiler_directives={'language_level' : "3"}),
)
