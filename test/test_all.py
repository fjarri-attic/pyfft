import optparse
import sys

from test_doc import run as run_doc
from test_performance import run as run_perf
from test_errors import run as run_err
from test_functionality import run as run_func
from helpers import DEFAULT_BUFFER_SIZE


# Parser settings

parser = optparse.OptionParser(usage = "test_all.py <mode> [options]\n" +
	"Modes: func, err, doc, perf")

parser.add_option("--cd", "--cuda", action="store_true",
	dest="test_cuda", help="run Cuda tests", default=False)
parser.add_option("--cl", "--opencl", action="store_true",
	dest="test_opencl", help="run OpenCL tests", default=False)

parser.add_option("-s", "--buffer_size", action="store", type="int", default=DEFAULT_BUFFER_SIZE,
	dest="buffer_size", help="Maximum test buffer size, Mb")

# Parse options and run tests

if len(sys.argv) == 1:
	parser.error("Error: mode should be specified")

# FIXME: find a way to do it using OptionParser
modes = ['func', 'err', 'doc', 'perf']
mode = sys.argv[1]
args = sys.argv[2:]

if mode.startswith("-"):
	args = [mode] + args
elif mode not in modes:
	parser.print_help()
	sys.exit(1)

opts, args = parser.parse_args(args)

if not opts.test_cuda and not opts.test_opencl:
	opts.test_cuda = isCudaAvailable()
	opts.test_opencl = isCLAvailable()

if mode == 'func':
	run_func(opts.test_cuda, opts.test_opencl)
elif mode == 'err':
	run_err(opts.test_cuda, opts.test_opencl, opts.buffer_size)
elif mode == 'doc':
	run_doc()
elif mode == 'perf':
	run_perf(opts.test_cuda, opts.test_opencl, opts.buffer_size)
