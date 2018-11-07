
from torch.utils.ffi import _wrap_function
from ._nms import lib as _lib, ffi as _ffi
import sys

__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        print("debug _import_symbols")
        print(symbol)
        fn = getattr(_lib, symbol)
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
        __all__.append(symbol)
    sys.stdout.flush()

_import_symbols(locals())
