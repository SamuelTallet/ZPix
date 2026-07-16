from logging import INFO, Formatter, StreamHandler, getLogger

logger = getLogger("ZPix")
"""Custom logger."""

logger.setLevel(INFO)

_log_handler = StreamHandler()
_log_handler.setFormatter(Formatter("%(levelname)s: %(message)s"))

logger.addHandler(_log_handler)
logger.propagate = False
