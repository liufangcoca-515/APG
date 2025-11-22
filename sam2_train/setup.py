from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 设置CUDA和NVIDIA库的路径
cuda_path = "/usr/local/cuda-12.5"
nvidia_path = "/home/liufang882/anaconda3/envs/SAMMED3D/lib/python3.9/site-packages/nvidia"

setup(
    name='sam2_cuda',
    ext_modules=[
        CUDAExtension(
            name='_C',
            sources=['csrc/connected_components.cu'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    f'-I{cuda_path}/include',
                    f'-I{os.path.join(nvidia_path, "cusparse/include")}',
                    f'-I{os.path.join(nvidia_path, "nvjitlink/include")}',
                    '-Xlinker=-rpath,' + os.path.join(nvidia_path, "cusparse/lib"),
                    '-Xlinker=-rpath,' + os.path.join(nvidia_path, "nvjitlink/lib")
                ]
            },
            library_dirs=[
                f'{cuda_path}/lib64',
                os.path.join(nvidia_path, "cusparse/lib"),
                os.path.join(nvidia_path, "nvjitlink/lib")
            ],
            libraries=['cusparse', 'nvJitLink'],
            runtime_library_dirs=[
                f'{cuda_path}/lib64',
                os.path.join(nvidia_path, "cusparse/lib"),
                os.path.join(nvidia_path, "nvjitlink/lib")
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 