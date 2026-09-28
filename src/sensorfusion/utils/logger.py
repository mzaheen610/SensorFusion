import os
import sys
import threading
from datetime import datetime


class DualStream:
    """
    Thread-safe stream wrapper that duplicates writes to both the original
    terminal stream and a target log file, flushing immediately.
    """
    def __init__(self, terminal_stream, log_file, lock):
        self.terminal = terminal_stream
        self.log_file = log_file
        self.lock = lock

    def write(self, message):
        with self.lock:
            try:
                self.terminal.write(message)
                self.terminal.flush()
            except Exception:
                pass
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except Exception:
                pass

    def flush(self):
        with self.lock:
            try:
                self.terminal.flush()
            except Exception:
                pass
            try:
                self.log_file.flush()
            except Exception:
                pass

    def fileno(self):
        return self.terminal.fileno()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()


_active_log_file = None
_orig_stdout = None
_orig_stderr = None


def setup_runtime_logger(logs_dir=None, prefix="run_", date_format="%Y_%m_%d_%H_%M_%S"):
    """
    Redirects sys.stdout and sys.stderr to simultaneously output to the terminal
    and append to a newly created, timestamped log file.

    Parameters
    ----------
    logs_dir : str or Path, optional
        Target directory for log files. Defaults to the repository 'logs/' directory.
    prefix : str, optional
        Prefix for the log filename (default: "run_").
    date_format : str, optional
        Strftime format for the timestamp (default: "%Y_%m_%d_%H_%M_%S").

    Returns
    -------
    str
        Absolute path to the created log file.
    """
    global _active_log_file, _orig_stdout, _orig_stderr

    # Avoid installing multiple times in the same process
    if _active_log_file is not None and not _active_log_file.closed:
        return _active_log_file.name

    if logs_dir is None:
        env_log_dir = os.environ.get("SENSORFUSION_LOG_DIR")
        if env_log_dir:
            logs_dir = os.path.abspath(env_log_dir)
        else:
            # Default to repo root / logs
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            logs_dir = os.path.join(repo_root, "logs")

    os.makedirs(logs_dir, exist_ok=True)

    env_log_file = os.environ.get("SENSORFUSION_LOG_FILE")
    if env_log_file:
        log_file_path = os.path.abspath(env_log_file)
    else:
        now_str = datetime.now().strftime(date_format)
        log_filename = f"{prefix}{now_str}.log"
        log_file_path = os.path.join(logs_dir, log_filename)

    _active_log_file = open(log_file_path, "a", encoding="utf-8", buffering=1)
    lock = threading.Lock()

    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr

    sys.stdout = DualStream(_orig_stdout, _active_log_file, lock)
    sys.stderr = DualStream(_orig_stderr, _active_log_file, lock)

    print(f"[Logger] Runtime logging started. Saving terminal prints to: {log_file_path}")
    return log_file_path


def teardown_runtime_logger():
    """
    Restores original sys.stdout and sys.stderr and closes the active log file.
    Useful for unit testing and cleanup.
    """
    global _active_log_file, _orig_stdout, _orig_stderr
    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
        _orig_stdout = None
    if _orig_stderr is not None:
        sys.stderr = _orig_stderr
        _orig_stderr = None
    if _active_log_file is not None and not _active_log_file.closed:
        try:
            _active_log_file.flush()
            _active_log_file.close()
        except Exception:
            pass
        _active_log_file = None
