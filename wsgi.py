from dashboard import _init_subsystems, create_app

# Patch MT5Connector.connect to always succeed on Linux
import mt5_connector
_orig_connect = mt5_connector.MT5Connector.connect
def _patched_connect(self):
    try:
        return _orig_connect(self)
    except Exception:
        return True
mt5_connector.MT5Connector.connect = _patched_connect

# Patch sys.exit to prevent shutdown on MT5 failure
import sys
_real_exit = sys.exit
def _no_exit(code=0):
    if code == 1:
        pass  # Suppress MT5 connection failure exit
    else:
        _real_exit(code)
sys.exit = _no_exit

_init_subsystems()
sys.exit = _real_exit  # Restore real exit after init

app = create_app()

if __name__ == "__main__":
    app.run()
