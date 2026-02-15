"""
Setup script for cuda_bargain package (in simulator folder).

Build with: pip install -e .
Or: python setup.py build_ext --inplace
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
here = os.path.dirname(os.path.abspath(__file__))

# CUDA architectures for Great Lakes
# sm_70 = V100 (gpu partition)
# sm_86 = A40 (spgpu partition)
cuda_archs = [
    '-gencode=arch=compute_70,code=sm_70',
    '-gencode=arch=compute_86,code=sm_86',
]

setup(
    name='cuda_bargain',
    version='0.1.0',
    description='CUDA-accelerated Bargaining Game Environment for Deep RL',
    author='SGRD Project',
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
