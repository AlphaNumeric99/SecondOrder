from __future__ import annotations

import os
from unittest.mock import patch

from app.services.env_safety import sanitize_ssl_keylogfile


def test_sanitize_ssl_keylogfile_unsets_unwritable_path():
    with patch.dict(os.environ, {"SSLKEYLOGFILE": r"Z:\does-not-exist\virtual_file.log"}, clear=False):
        with patch("builtins.open", side_effect=PermissionError):
            sanitize_ssl_keylogfile()
        assert "SSLKEYLOGFILE" not in os.environ


def test_sanitize_ssl_keylogfile_keeps_usable_path():
    with patch.dict(os.environ, {"SSLKEYLOGFILE": r"C:\tmp\keylog.log"}, clear=False):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open"):
                sanitize_ssl_keylogfile()
        assert os.environ.get("SSLKEYLOGFILE") == r"C:\tmp\keylog.log"
