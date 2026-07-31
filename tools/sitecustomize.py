# tools/sitecustomize.py — auto-imported at interpreter startup
import ssl

_orig = ssl.SSLContext._load_windows_store_certs

def _safe(self, storename, purpose):
    try:
        return _orig(self, storename, purpose)
    except (ssl.SSLError, OSError):
        return bytearray()   # skip an unparseable cert store instead of crashing

ssl.SSLContext._load_windows_store_certs = _safe