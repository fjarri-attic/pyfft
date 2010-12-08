r"""
===========================
FFT for PyCuda and PyOpenCL
===========================

.. contents::

Introduction
------------

This module contains implementation of batched FFT, ported from `Apple's OpenCL implementation
<https://developer.apple.com/mac/library/samplecode/OpenCL_FFT/index.html>`_.
OpenCL's ideology of constructing kernel code on the fly maps perfectly on
`PyCuda <http://mathema.tician.de/software/pycuda>`_/`PyOpenCL <http://mathema.tician.de/software/pyopencl>`_,
and variety of Python's templating engines makes code generation simpler. I used
`mako <http://pypi.python.org/pypi/Mako>`_ templating engine, simply because of
the personal preference. The code can be easily changed to use any other engine.

**Warning**: "Cuda" part of ``pyfft`` requires ``PyCuda 0.94`` or newer;
"CL" part requires ``PyOpenCL 0.92`` or newer.

Quick Start
-----------

This overview contains basic usage examples for both backends, CUDA and OpenCL.
CUDA part goes first and contains a bit more detailed comments,
but they can be easily projected on OpenCL part, since the code is very similar.

~~~~~~~~~~~~
CUDA version
~~~~~~~~~~~~

First, import ``numpy`` and plan creation interface from ``pyfft``.

 >>> from pyfft.cuda import Plan
 >>> import numpy

Since we are using Cuda, it must be initialized before any Cuda functions are called
(by default, the plan will use existing context, but there are other possibilities;
see reference entry for `Plan` for further information).
In addition, we will need ``gpuarray`` module to pass data to and from GPU.
Stream creation is optional; if no stream is provided, ``Plan`` will create its own one.

 >>> from pycuda.tools import make_default_context
 >>> import pycuda.gpuarray as gpuarray
 >>> import pycuda.driver as cuda
 >>> cuda.init()
 >>> context = make_default_context()
 >>> stream = cuda.Stream()

Then the plan must be created.
The creation is not very fast, mainly because of the compilation speed.
But, fortunately, ``PyCuda`` and ``PyOpenCL`` cache compiled sources,
so if you use the same plan for each run of your program, it will be compiled only the first time.

 >>> plan = Plan((16, 16), stream=stream)

Now, let's prepare simple test array:

 >>> data = numpy.ones((16, 16), dtype=numpy.complex64)
 >>> gpu_data = gpuarray.to_gpu(data)
 >>> print gpu_data # doctest: +ELLIPSIS
 [[ 1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j
    1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j]
 ...
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j
    1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j  1.+0.j]]

... and execute our plan:

 >>> plan.execute(gpu_data) # doctest: +ELLIPSIS
 <pycuda._driver.Stream object at ...>
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
Let's now perform the inverse transform:

 >>> plan.execute(gpu_data, inverse=True) # doctest: +ELLIPSIS
 <pycuda._driver.Stream object at ...>
 >>> result = gpu_data.get()

Since data is non-integer, we cannot simply compare it. We will just calculate error instead.

 >>> error = numpy.abs(numpy.sum(numpy.abs(data) - numpy.abs(result)) / data.size)
 >>> error < 1e-6
 True

That's good enough for single precision numbers.

Last step is releasing Cuda context:

 >>> context.pop()

Reference
---------

.. _Plan:

.. _cuda.Plan:

.. _cl.Plan:

cuda.Plan, cl.Plan
~~~~~~~~~~~~~~~~~~

Creates class, containing precalculated FFT plan.

**Arguments**: ``Plan(shape, dtype=numpy.complex64, mempool=None, context=None, normalize=True,
wait_for_finish=None, fast_math=True, stream=None, queue=None)``

``shape``:
  Problem size. Can be integer or tuple with 1, 2 or 3 integer elements. Each dimension must be
  a power of two.

  **Warning**: 2D and 3D plans with ``y`` == 1 or ``z`` == 1 are not supported at the moment.

``dtype``:
  Numpy data type for input/output arrays. If complex data type is given, plan for interleaved
  arrays will be created. If scalar data type is given, plan will work for data arrays with
  separate real and imaginary parts. Depending on this parameter, `execute()`_ will have
  different signatures; see its reference entry for details.

  *Currently supported*: ``numpy.complex64``, ``numpy.float32`` (single precision) and
  ``numpy.complex128``, ``numpy.float64`` (double precision).

``mempool``:
  **Cuda-specific**. If specified, method ``allocate`` of this object will be used to create
  temporary buffers.

``normalize``:
  Whether to normalize inverse FFT so that IFFT(FFT(signal)) == signal. If equals to ``False``,
  IFFT(FFT(signal)) == signal * x * y * z.

``wait_for_finish``:
  Boolean variable, which tells whether it is necessary to wait on stream after scheduling all
  FFT kernels. Default value depends on ``context``, ``stream`` and ``queue`` parameters --- see
  `Contexts and streams usage logic`_ for details. Can be overridden by ``wait_for_finish`` parameter
  to `execute()`_

``fast_math``:
  If ``True``, additional compiler options will be used, which increase performance at the expense of
  accuracy. For **Cuda** it is ``-use_fast_math``, for **OpenCL** --- ``-cl-mad-enable`` and
  ``-cl-fast-relaxed-math``. In addition, in case of **OpenCL**, ``native_cos`` and ``native_sin``
  are used instead of ``cos`` and ``sin`` (**Cuda** uses intrinsincs automatically when
  ``-use_fast_math`` is set).

``context``:
  Context, which will be used to compile kernels and execute plan. See `Contexts and streams usage logic`_
  entry for details.

``stream``:
  **Cuda-specific**. An object of class ``pycuda.driver.Stream``, which will be used to schedule
  plan execution.

``queue``:
  **OpenCL-specific**. An object of class ``pyopencl.CommandQueue``, which will be used to schedule
  plan execution.

.. _execute():

Plan.execute()
~~~~~~~~~~~~~~

Execute plan for interleaved data arrays. Signature depends on ``dtype`` given to constructor:

**Interleaved**: ``execute(data_in, data_out=None, inverse=False, batch=1, wait_for_finish=None)``

**Split**: ``executeSplit(data_in_re, data_in_im, data_out_re=None, data_out_im=None,
inverse=False, batch=1, wait_for_finish=None)``

``data_in`` or ``data_in_re``, ``data_in_im``:
  Input array(s). For Cuda plan PyCuda's ``GPUArray`` or anything that can be cast to memory pointer
  is supported; for OpenCL ``Buffer`` objects are supported.

``data_out`` or ``data_out_re``, ``data_out_im``:
  Output array(s). If not defined, the execution will be performed in-place and the results
  will be stored in ``data_in`` or ``data_in_re``, ``data_in_im``.

``inverse``:
  If ``True``, inverse transform will be performed.

``batch``:
  Number of data sets to process. They should be located successively in ``data_in``.

``wait_for_finish``:
  Whether to wait for scheduled FFT kernels to finish. Overrides setting, which was specified
  during plan creation.

**Returns**
  ``None`` if waiting for scheduled kernels; ``Stream`` or ``CommandQueue`` object otherwise.
  User is expected to handle this object with care, since it can be reused during the next call
  to `execute()`_.

Contexts and streams usage logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plan behavior can differ depending on values of ``context``, ``stream``/``queue`` and
``wait_for_finish`` parameters. These differences should, in theory, make the module
more convenient to use.

``wait_for_finish`` parameter can be set on three levels. First, there is a default value
which depends on ``context`` and ``stream``/``queue`` parameters (see details below). It
can be overridden by explicitly passing it as an argument to constructor. This setting,
in turn, can be overridden by passing ``wait_for_finish`` keyword to `execute()`_.

----
Cuda
----

1. ``context`` and ``stream`` are ``None``:

  * Current (at the moment of plan creation) context and device will be used to create kernels.
  * ``Stream`` will be created internally and used for each `execute()`_ call.
  * Default value of ``wait_for_finish`` is ``True``.

2. ``stream`` is not ``None``:

  * ``context`` is ignored.
  * ``stream`` is remembered and used.
  * `execute()`_ will assume that context, corresponding to given stream is active at the time of the call.
  * Default value of ``wait_for_finish`` is ``False``.

3. ``context`` is not ``None``:

  * `execute()`_ will assume that context, corresponding to given one is active at the time of the call.
  * New ``Stream`` is created each time `execute()`_ is called and destroyed if ``wait_for_finish``
    finally evaluates to ``True``.
  * Default value of ``wait_for_finish`` is ``True``.

------
OpenCL
------

Either ``context`` or ``queue`` must be set.

1. ``queue`` is not ``None``:

  * ``queue`` is remembered and used.
  * Target context and device are obtained from ``queue``.
  * Default value of ``wait_for_finish`` is ``False``.

2. ``context`` is not ``None``:

  * ``context`` is remembered.
  * ``CommandQueue`` will be created internally and used for each `execute()`_ call.
  * Default value of ``wait_for_finish`` is ``True``.

Performance
~~~~~~~~~~~

Here is the comparison to pure Cuda program using CUFFT. Since CUFFT supports batched
FFT only for 1D, tests for other dimensions were performed using 1D FFT and matrix
transposes (that's why CUFFT performance is so non-uniform, but it
is the only way of getting FFT for large batches). See ``cuda`` folder in distribution
for details. Pyfft tests were executed with ``fast_math==True``.

In the following tables "sp" stands for "single precision", "dp" for "double precision".

Mac OS 10.6.4, Python 2.6, Cuda 3.1, PyCuda 0.94, GF9600, 32 Mb buffer:

+---------------------------+------------+------------+
| Problem size / GFLOPS     | CUFFT, sp  | pyfft, sp  |
+===========================+============+============+
| [16, 1, 1], batch 131072  | 1.61       | 7.63       |
+---------------------------+------------+------------+
| [1024, 1, 1], batch 2048  | 16.54      | 15.71      |
+---------------------------+------------+------------+
| [8192, 1, 1], batch 256   | 13.13      | 12.25      |
+---------------------------+------------+------------+
| [16, 16, 1], batch 8192   | 1.65       | 10.04      |
+---------------------------+------------+------------+
| [128, 128, 1], batch 128  | 15.08      | 15.14      |
+---------------------------+------------+------------+
| [1024, 1024, 1], batch 2  | 14.80      | 13.38      |
+---------------------------+------------+------------+
| [16, 16, 16], batch 512   | 1.77       | 11.42      |
+---------------------------+------------+------------+
| [32, 32, 128], batch 16   | 5.32       | 15.65      |
+---------------------------+------------+------------+
| [128, 128, 128], batch 1  | 12.27      | 14.53      |
+---------------------------+------------+------------+

CentOS 5.5, Python 2.6, Cuda 3.1, PyCuda 0.94, nVidia Tesla C1060, 32 Mb buffer:

+---------------------------+------------+------------+------------+------------+
| Problem size / GFLOPS     | CUFFT, sp  | pyfft, sp  | CUFFT, dp  | pyfft, dp  |
+===========================+============+============+============+============+
| [16, 1, 1], batch 131072  | 99.83      | 81.17      | 27.16      | 11.02      |
+---------------------------+------------+------------+------------+------------+
| [1024, 1, 1], batch 2048  | 206.27     | 185.73     | 38.05      | 11.98      |
+---------------------------+------------+------------+------------+------------+
| [8192, 1, 1], batch 256   | 132.84     | 116.53     | 35.60      | 11.59      |
+---------------------------+------------+------------+------------+------------+
| [16, 16, 1], batch 8192   | 4.97       | 90.04      | 3.14       | 14.04      |
+---------------------------+------------+------------+------------+------------+
| [128, 128, 1], batch 128  | 71.03      | 127.53     | 22.64      | 15.80      |
+---------------------------+------------+------------+------------+------------+
| [1024, 1024, 1], batch 2  | 49.71      | 122.26     | 17.45      | 11.75      |
+---------------------------+------------+------------+------------+------------+
| [16, 16, 16], batch 512   | 6.77       | 93.40      | 4.17       | 15.25      |
+---------------------------+------------+------------+------------+------------+
| [32, 32, 128], batch 16   | 25.66      | 127.94     | 11.02      | 14.85      |
+---------------------------+------------+------------+------------+------------+
| [128, 128, 128], batch 1  | 52.26      | 115.22     | 16.35      | 15.30      |
+---------------------------+------------+------------+------------+------------+
"""

import doctest
import time
import sys

DOCUMENTATION = __doc__

def run():
	print "Running doctest..."
	doctest.testmod(m=sys.modules.get(__name__))

if __name__ == "__main__":
	time1 = time.time()
	run()
	time2 = time.time()
	print("Finished in {0:.3f} seconds".format(time2 - time1))
