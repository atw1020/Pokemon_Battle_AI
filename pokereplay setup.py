from distutils.core import setup, Extension


hello_module = Extension('pokereplay', sources=['pokereplay.cpp'])

setup(name='pokereplay',
      version='0.2.0',
      description='Hello world module written in C++',
      ext_modules=[hello_module])
