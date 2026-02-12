from __future__ import annotations

import os
from pathlib import Path


def sanitize_ssl_keylogfile() -> None:
    """Unset SSLKEYLOGFILE when it points to an unusable path.

    Some environments set this globally for TLS debugging. If the path is
    inaccessible, underlying HTTP clients can crash while creating SSL context.
    """
    keylog_path = os.getenv("SSLKEYLOGFILE", "").strip()
    if not keylog_path:
        return

    try:
        path = Path(keylog_path)
        parent = path.parent
        if parent and not parent.exists():
            os.environ.pop("SSLKEYLOGFILE", None)
            return

        # Validate writability without truncating existing files.
        with open(path, "a", encoding="utf-8"):
            pass
    except Exception:
        os.environ.pop("SSLKEYLOGFILE", None)
