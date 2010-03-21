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

Quick Start
-----------

The usage is quite simple. First, import ``numpy`` and plan creation interface from ``pyfft``
(let us use cuda in this example):

 >>> from pyfft.cuda import Plan
 >>> import numpy

Since we are using Cuda, it must be initialized before any Cuda functions are called
(by default, the plan will use existing context, but there are other possibilities;
see reference entry for `Plan` for further information). In addition, we will
need ``gpuarray`` module to pass data to and from GPU:

 >>> from pycuda.tools import make_default_context
 >>> import pycuda.gpuarray as gpuarray
 >>> import pycuda.driver as cuda
 >>> cuda.init()
 >>> context = make_default_context()

Then the plan must be created. The creation is not very fast, mainly because of the
compilation speed. But, fortunately, ``PyCuda`` and ``PyOpenCL`` cache compiled sources, so if you
use the same plan for each run of your program, it will be created pretty fast.

 >>> plan = Plan((16, 16))

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
Let's now perform the inverse transform:

 >>> plan.execute(gpu_data, inverse=True)
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
wait_for_finish=None, stream=None, queue=None)``

``shape``:
  Problem size. Can be integer or tuple with 1, 2 or 3 integer elements. Each dimension must be
  a power of two.

  **Warning**: 2D and 3D plans with ``y`` == 1 or ``z`` == 1 are not supported at the moment.

``dtype``:
  Numpy data type for input/output arrays. If complex data type is given, plan for interleaved
  arrays will be created. If scalar data type is given, plan will work for data arrays with
  separate real and imaginary parts. Depending on this parameter, `execute()`_ will have
  different signatures; see its reference entry for details.

  *Currently supported*: ``numpy.complex64`` and ``numpy.float32``.

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

1. ``context`` and ``stream`` are ``None``:

  * Context will be created using ``pyopencl.create_some_context()``, and its default device
    will be used to compile and execute kernels.
  * ``CommandQueue`` will be created internally and used for each `execute()`_ call.
  * Default value of ``wait_for_finish`` is ``True``.

2. ``queue`` is not ``None``:

  * ``queue`` is remembered and used.
  * Target context and device are obtained from ``queue``.
  * Default value of ``wait_for_finish`` is ``False``.

3. ``context`` is not ``None``:

  * ``context`` is remembered.
  * New ``CommandQueue`` will be created on remembered context's default device each time
    `execute()`_ is called and destroyed if ``wait_for_finish`` finally evaluates to ``False``.
  * Default value of ``wait_for_finish`` is ``True``.

Performance
~~~~~~~~~~~

Here is the comparison to PyCuda CUFFT wrapper. Results were obtained on Mac OS 10.6.2,
Python 2.6, Cuda 2.3, PyCuda 0.94, GF9400.

+---------------------------+------------+------------+
| Problem size / GFLOPS     | CUFFT      | pycudafft  |
+===========================+============+============+
| [16, 1, 1], batch 131072  | 1.06       | 4.50       |
+---------------------------+------------+------------+
| [1024, 1, 1], batch 2048  | 9.32       | 7.54       |
+---------------------------+------------+------------+
| [8192, 1, 1], batch 256   | 9.27       | 6.24       |
+---------------------------+------------+------------+
| [16, 16, 1], batch 8192   | 0.81       | 6.22       |
+---------------------------+------------+------------+
| [128, 128, 1], batch 128  | 8.58       | 7.81       |
+---------------------------+------------+------------+
| [1024, 1024, 1], batch 2  | 7.71       | 6.72       |
+---------------------------+------------+------------+
| [16, 16, 16], batch 512   | 0.85       | 6.93       |
+---------------------------+------------+------------+
| [32, 32, 128], batch 16   | 2.60       | 7.46       |
+---------------------------+------------+------------+
| [128, 128, 128], batch 1  | 6.37       | 7.60       |
+---------------------------+------------+------------+

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
