"""Install T25/TL patches only for explicitly marked training processes."""

import os


if os.environ.get("REGRESSION_SAFE_TAIL_ENABLED", "false").lower() == "true":
    from regression_safe_tail_length_runtime_patch import install

    install()
