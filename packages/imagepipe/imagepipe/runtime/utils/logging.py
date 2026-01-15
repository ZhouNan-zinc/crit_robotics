from __future__ import annotations

from typing import Any, Callable, Protocol, TYPE_CHECKING
from typing import Any
from types import ModuleType


LogFn = Callable[..., None]

class LoggerLike(Protocol):
    def debug(self, msg: object, *args: Any, **kwargs: Any) -> None: ...
    def info(self, msg: object, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, msg: object, *args: Any, **kwargs: Any) -> None: ...
    def error(self, msg: object, *args: Any, **kwargs: Any) -> None: ...
    def exception(self, msg: object, *args: Any, **kwargs: Any) -> None: ...
    def fatal(self, msg: object, *args: Any, **kwargs: Any) -> None: ...

debug: LogFn
info: LogFn
warning: LogFn
error: LogFn
exception: LogFn
fatal: LogFn

getLogger: Callable[[str], LoggerLike]
ROOT_LOGGER: LoggerLike

def set_logging_backend(backend: str|ModuleType) -> None:
    """
    Dynamically configure logging backend.

    Configures the module in-place so `logging_module` can be used consistently.
    """
    
    global debug, info, warning, error, exception, fatal
    global getLogger
    global ROOT_LOGGER

    if isinstance(backend, ModuleType):
        set_logging_backend(getattr(backend, "__name__"))

    if backend == "logging":
        import logging
        # logging.basicConfig(level=logging.INFO)

        debug = logging.debug
        info = logging.info
        warning = logging.warning
        error = logging.error
        exception = logging.exception
        fatal = logging.fatal

        getLogger = logging.getLogger


    elif backend == "rclpy.logging" or backend == "rclpy":
        try:
            import rclpy.logging
        except ImportError:
            raise ImportError("rclpy is not available.")
        
        # NOTE: For rclpy, name is specified as node name.
        getLogger = rclpy.logging.get_logger("imagepipe")

    else:
        raise TypeError(f"Unsupported logging backend: {backend!r}")

set_logging_backend("logging")