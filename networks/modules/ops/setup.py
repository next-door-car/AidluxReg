from setuptools import find_packages, setup
import os
import subprocess
import sys
import time
import torch
from torch.utils.cpp_extension import (BuildExtension, 
                                       CppExtension,
                                       CUDAExtension)
version_file = 'version.py' # 当前文件的目录 version.py 中 __version__='0.0.1' 保存文件格式为UTF-8

def get_git_hash():
    '''
    含义是获取git的hash值
    '''
    def _minimal_ext_cmd(cmd): # 它接收一个命令列表作为参数
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha

def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from version import __version__ 
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha
def get_readme(filename='README.md'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), encoding='utf-8') as f:
        content = f.read()
    return content
def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__)) # 含义是当前文件所在的目录
    with open(os.path.join(here, filename), mode='r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires
def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
short_version = '{}'
version_info = ({})
""" # 不能修改其格式
    sha = get_hash() # 含义是获取当前git的commit id
    with open('VERSION', 'r') as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = ', '.join(
        [x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split('.')])
    VERSION = SHORT_VERSION + '+' + sha

    version_file_str = content.format(time.asctime(), VERSION, SHORT_VERSION,
                                      VERSION_INFO)
    with open(version_file, 'w') as f:
        f.write(version_file_str)

def make_cuda_ext(name, module, sources, sources_cuda=None):
    if sources_cuda is None:
        sources_cuda = []
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    
if __name__ == '__main__':
    if '--no_cuda_ext' in sys.argv: # sys.argv 是 一个列表，包含命令行参数
        ext_modules = []
        sys.argv.remove('--no_cuda_ext')
    else:
        ext_modules = [
            make_cuda_ext(
                name='deform_conv_ext',
                module='dcn',
                sources=['src/deform_conv_ext.cpp'],
                sources_cuda=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu'
                ]
            )
        ]

    write_version_py()
    setup(
        name='basicops',
        version=get_version(),
        description='Open Source Image and Video Toolbox',
        long_description=get_readme(),
        author='Zhong Hao',
        author_email='Hownzcc0792@gmail.com',
        keywords='computer vision',
        url='https://github.com/Hownz/WJ-AI',
        packages=find_packages(
            exclude=()
        ),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        setup_requires=['cython', 'numpy'],
        install_requires=get_requirements(),
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
