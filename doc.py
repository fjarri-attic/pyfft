r"""
==============
FFT for PyCuda
==============

.. contents::

Introduction
------------

At the moment of writing this, PyCuda experienced problems with CUFFT caused by nVidia's
architectural decisions. As a result, this piece of software was written. It contains
implementation of batched FFT, ported from `Apple's OpenCL implementation
<https://developer.apple.com/mac/library/samplecode/OpenCL_FFT/index.html>`_.
OpenCL's ideology of constructing kernel code on the fly maps perfectly on PyCuda,
and variety of Python's templating engines makes code generation simpler. I used
`mako <http://pypi.python.org/pypi/Mako>`_ templating engine, simply because of
the personal preference. The code can be easily changed to use any other engine.

In addition, project repository contains wrapper, which adds batch support for CUFFT and
its mapping to PyCuda. They were added for testing purposes and will not be actively
supported.

Quick Start
-----------

The usage is quite simple. First, import ``pycudafft`` and ``numpy``:

 >>> import pycuda.autoinit
 >>> import pycuda.gpuarray as gpuarray
 >>> from pycudafft import FFTPlan
 >>> import numpy

Then the plan must be created. The creation is not very fast, mainly because of the
compilation speed. But, fortunately, ``PyCuda`` caches compiled sources, so if you
use the same plan for each run of your program, it will be created pretty fast.

 >>> plan = FFTPlan(16, 16)

Now, let's prepare simple test array and try to execute plan over it:

 >>> data = numpy.ones((16, 16), dtype=numpy.complex64)
 >>> gpu_data = gpuarray.to_gpu(data)
 >>> plan.execute(gpu_data)
 >>> result = gpu_data.get()
 >>> print result # doctest: +ELLIPSIS
 [[ 256.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j
      0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j
      0.+0.j    0.+0.j]
 ...
  [   0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j
      0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j    0.+0.j
      0.+0.j    0.+0.j]]

As expected, we got array with the first non-zero element, equal to array size.
Let's now perform the inverse transform. Notice the normalization.

 >>> plan.execute(gpu_data, inverse=True)
 >>> result = gpu_data.get() / data.size

Since data is non-integer, we cannot simply compare it. We will just calculate error instead.

 >>> error = numpy.abs(numpy.sum(numpy.abs(data) - numpy.abs(result)) / data.size)
 >>> error < 1e-6
 True

That's good enough for single precision numbers.

Reference
---------

FFTPlan
~~~~~~~

Class, containing precalculated FFT plan.

**Arguments**: ``FFTPlan(x, y=None, z=None, split=False, dtype=numpy.complex64, mempool=None)``

``x``, ``y``, ``z``:
  Problem size. If ``z`` and/or ``y`` are not defined, 2D or 1D plan will be created.

  **Warning**: 2D and 3D plans with ``y`` == 1 or ``z`` == 1 are not supported at the moment.

``split``:
  If ``True``, the plan will support separate data arrays with real and imaginary parts
  instead of interleaved arrays.

``dtype``:
  Data type to use. At the moment only ``complex64`` (single precision) is supported.

``mempool``:
  If specified, method ``allocate`` of this object will be used to create temporary buffers.

FFTPlan.execute()
~~~~~~~~~~~~~~~~~

Execute plan for interleaved data arrays.

**Arguments**: ``execute(data_in, data_out=None, inverse=False, batch=1)``

``data_in``:
  Input array.

``data_out``:
  Output array. If not defined, the execution will be performed in-place and the results
  will be stored in ``data_in``.

``inverse``:
  If ``True``, inverse transform will be performed.

``batch``:
  Number of data sets to process. They should be located successively in ``data_in``.

FFTPlan.executeSplit()
~~~~~~~~~~~~~~~~~~~~~~

Execute plan for split data arrays.

**Arguments**: ``executeSplit(data_in_re, data_in_im, data_out_re=None, data_out_im=None, inverse=False, batch=1)``

``data_in_re``, ``data_in_im``:
  Input arrays with real and imaginary data parts correspondingly.

``data_out_re``, ``data_out_im``:
  Output arrays. If not defined, the execution will be performed in-place and the results
  will be stored in ``data_in_re`` and ``data_in_im``.

``inverse``:
  If ``True``, inverse transform will be performed.

``batch``:
  Number of data sets to process. They should be located successively in ``data_in``.

Performance
~~~~~~~~~~~

Here is the comparison to PyCuda CUFFT wrapper. Results were obtained on Mac OS 10.6.2,
Python 2.6, Cuda 2.3, PyCuda 0.94, GF9400.

+---------------------------+------------+------------+
| Problem size / GFLOPS     | CUFFT      | pycudafft  |
+===========================+============+============+
| [16, 1, 1], batch 131072  | 1.06       | 1.34       |
+---------------------------+------------+------------+
| [1024, 1, 1], batch 2048  | 9.32       | 1.21       |
+---------------------------+------------+------------+
| [8192, 1, 1], batch 256   | 9.27       | 1.26       |
+---------------------------+------------+------------+
| [16, 16, 1], batch 8192   | 0.81       | 1.91       |
+---------------------------+------------+------------+
| [128, 128, 1], batch 128  | 8.58       | 2.18       |
+---------------------------+------------+------------+
| [1024, 1024, 1], batch 2  | 7.71       | 1.29       |
+---------------------------+------------+------------+
| [16, 16, 16], batch 512   | 0.85       | 2.27       |
+---------------------------+------------+------------+
| [32, 32, 128], batch 16   | 2.60       | 2.48       |
+---------------------------+------------+------------+
| [128, 128, 128], batch 1  | 6.37       | 2.61       |
+---------------------------+------------+------------+

"""

import doctest
import time
import sys

DOCUMENTATION = __doc__

if __name__ == "__main__":
	print("=" * 70)
	time1 = time.time()
	doctest.testmod(m=sys.modules.get(__name__))
	time2 = time.time()
	print("=" * 70)

	print("Finished in {0:.3f} seconds".format(time2 - time1))
