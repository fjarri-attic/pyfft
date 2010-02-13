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

 >>> from pycudafft import FFTPlan
 >>> import numpy

Then the plan must be created. The creation is not very fast, mainly because of the
compilation speed. But, fortunately, ``PyCuda`` caches compiled sources, so if you
use the same plan for each run of your program, it will be created pretty fast.

 >>> plan = FFTPlan(16, 16)

Now, let's prepare simple test array and try to execute plan over it:
"""

import doctest
import time
import sys

DOCUMENTATION = __doc__

if __name__ == "__main__":
	print("=" * 70)
	time1 = time.time()
	doctest.testmod(m=sys.modules.get(__name__), verbose=False if verbosity < 3 else True)
	time2 = time.time()
	print("=" * 70)

	print("Finished in {0:.3f} seconds".format(time2 - time1))
