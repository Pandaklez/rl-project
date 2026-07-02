"""
Compatibility shim for `from config import cfg` when `config` is a directory.

This file loads the sibling `config.py` module (which defines `cfg`) and
exposes `cfg` at package import time so existing imports like
`from config import cfg` continue to work when running from inside
`smpler-x-main`.
"""
import importlib.util
import os

_here = os.path.dirname(__file__)
_config_py = os.path.abspath(os.path.join(_here, '..', 'config.py'))

spec = importlib.util.spec_from_file_location('smplerx_config_py', _config_py)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

# expose cfg symbol for `from config import cfg`
cfg = getattr(_mod, 'cfg')
