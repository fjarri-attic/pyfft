try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os.path

from doc import DOCUMENTATION

VERSION = '0.1'

# generate .rst file with documentation
open(os.path.join(os.path.dirname(__file__), 'documentation.rst'), 'w').write(DOCUMENTATION)

setup(
	name='pycudafft',
	packages=['pycudafft', 'cufft'],
	requires=['pycuda', 'mako', 'numpy'],
	version=VERSION,
	author='Bogdan Opanchuk',
	author_email='bg@bk.ru',
	url='http://github.com/Manticore/pycudafft',
	description='FFT library for PyCuda',
	long_description=DOCUMENTATION,
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Developers',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 2',
		'Topic :: Scientific/Engineering :: Mathematics'
	]
)
