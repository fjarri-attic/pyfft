PyFFT is a module containing Apple's FFT implementation ported for PyCuda and PyOpenCL.
Documentation can be found in `doc`, managed by `Sphinx <http://sphinx.pocoo.org>`_
(or, in rendered form, on `project's page <http://packages.python.org/pyfft>`_).

Module will work even if only one of PyCuda and PyOpenCL is present; that's why they are not
added to dependencies, and it is your responsibility to install the module you need.
