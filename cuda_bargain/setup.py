"""
Setup script for cuda_bargain package.

Build with: pip install -e .
Or: python setup.py build_ext --inplace
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
here = os.path.dirname(os.path.abspath(__file__))

# CUDA architectures
# sm_89 = RTX 4090 (Ada Lovelace)
# sm_90 = H100 (Hopper)
# sm_86 = RTX 3090 (Ampere)
# sm_80 = A100 (Ampere)
cuda_archs = [
    '-gencode=arch=compute_80,code=sm_80',
    '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_89,code=sm_89',
    '-gencode=arch=compute_90,code=sm_90',
]

setup(
    name='cuda_bargain',
    version='0.1.0',
    description='CUDA-accelerated Bargaining Game Environment for Deep RL',
    author='Causal Game Analysis Project',
    packages=['cuda_bargain'],
    package_dir={'cuda_bargain': 'python'},
    ext_modules=[
        CUDAExtension(
            name='cuda_bargain.cuda_bargain_core',
            sources=[
                'src/bargain_game.cu',
                'src/python_bindings.cpp',
            ],
            include_dirs=[
                os.path.join(here, 'include'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                ] + cuda_archs,
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0',
        'numpy',
    ],
    extras_require={
        'test': [
            'pytest',
        ],
    },
)
