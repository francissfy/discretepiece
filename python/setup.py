#!/usr/bin/env python

# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.!

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
import codecs
import string
import subprocess
import sys
import os


def long_description():
  with codecs.open('../README.md', 'r', 'utf-8') as f:
    long_description = f.read()
  return long_description


def get_cflags_and_libs(root):
  cflags = [
    '-std=c++17', 
    '-Wno-deprecated-declarations',  # for sprintf issue in _wrap.cxx
    '-I' + os.path.join(root, '../../..'),
    '-I' + os.path.join(root, '../../../src'),
    '-I' + os.path.join(root, '../../../src/builtin_pb'),
  ]
  libs = []
  if os.path.exists(os.path.join(root, 'lib/pkgconfig/discretepiece.pc')):
    libs = [
        os.path.join(root, 'lib/libdiscretepiece_encode.a'),
        os.path.join(root, 'lib/libdiscretepiece_train.a'),
    ]
  return cflags, libs


class build_ext(_build_ext):
  """Override build_extension to run cmake."""

  def build_extension(self, ext):
    cflags, libs = get_cflags_and_libs('./build/root')

    if len(libs) == 0:
        subprocess.check_call(['./build_bundled.sh'])
        cflags, libs = get_cflags_and_libs('./build/root')

    # Fix compile on some versions of Mac OSX
    # See: https://github.com/neulab/xnmt/issues/199
    if sys.platform == 'darwin':
      cflags.append('-mmacosx-version-min=14.2')
    else:
      cflags.append('-Wl,-strip-all')
      libs.append('-Wl,-strip-all')
    print('## cflags={}'.format(' '.join(cflags)))
    print('## libs={}'.format(' '.join(libs)))
    ext.extra_compile_args = cflags
    ext.extra_link_args = libs
    _build_ext.build_extension(self, ext)


SPM_EXT = Extension(
    'discretepiece._discretepiece',
    sources=['src/discretepiece/spm_client_wrap.cxx'],
)
cmdclass = {'build_ext': build_ext}


setup(
  name='discretepiece',
  author='Feiyu Shen',
  author_email='francis_sfy@sjtu.edu.cn',
  description='DiscretePiece Client python wrapper',
  long_description=long_description(),
  long_description_content_type='text/markdown',
  version=1.00,
  license='Apache',
  platforms='Unix',
  ext_modules=[SPM_EXT],
  package_dir={'': 'src'},
  py_modules=[
    'discretepiece/__init__',
    'discretepiece/discretepiece'
  ],
  cmdclass=cmdclass,
)
