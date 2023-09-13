from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='libnway',
      packages=find_packages(),
      ext_modules=[
          cpp_extension.CppExtension('libnway', [
              'clib/nway-align.cpp'
            ])
        ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

