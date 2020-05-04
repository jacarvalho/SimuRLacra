import os
import os.path as osp


# Global log prefix directory
_log_prefix_dir = None


def set_log_prefix_dir(log_dir):
    """ Set the log prefix directory for this session. """
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    else:
        assert osp.isdir(log_dir)

    global _log_prefix_dir
    _log_prefix_dir = log_dir


def get_log_prefix_dir():
    """ Get the log prefix directory for this session. """
    return _log_prefix_dir


def resolve_log_path(path):
    """ Resolve path against the global log dir if path is relative. """
    if _log_prefix_dir is not None:
        # Join properly handles absolute path
        path = osp.join(_log_prefix_dir, path)
    return path
