"""Logging utilities for the SwarmRL package."""

import sys
import typing

import jax
from loguru import logger


class _SwarmRLLoggingConfig:
    """Internal mutable logging settings shared across helper functions."""

    def __init__(self) -> None:
        self.sink_ids: typing.List[int] = []
        self.jax_runtime_log_level_no: int = logger.level("DEBUG").no
        self.jax_runtime_log_enabled: bool = True


_LOGGING_CONFIG = _SwarmRLLoggingConfig()


def _to_level_number(level: typing.Union[int, str]) -> int:
    """Convert level names or numeric values to a numeric logging level."""

    if isinstance(level, str):
        return logger.level(level.upper()).no
    return int(level)


def set_jax_runtime_log_level(level: typing.Union[int, str]) -> None:
    """Set callback registration threshold for JAX runtime value logging."""

    _LOGGING_CONFIG.jax_runtime_log_level_no = _to_level_number(level)


def set_jax_runtime_log_enabled(enabled: bool) -> None:
    """Enable or disable JAX runtime value logging callbacks globally."""

    _LOGGING_CONFIG.jax_runtime_log_enabled = bool(enabled)


def setup_swarmrl_logger(
    filename: typing.Optional[str] = None,
    loglevel_terminal: typing.Union[int, str] = "INFO",
    loglevel_file: typing.Union[int, str] = "DEBUG",
    include_user_logs: bool = False,
    remove_default_sink: bool = True,
    log_jax_values: bool = False,
):
    """
    Configure package logging with Loguru and enable swarmrl log output.

    This function is opt-in and is intended to be called from user scripts.
    Before calling it, swarmrl logging is disabled by default in ``swarmrl.__init__``.

    Parameters
    ----------
    filename
            Name of the file where logs get written to. If None or an empty string,
            no file sink is created.
    loglevel_terminal
            Terminal log level for Loguru sinks. Supports Loguru level names
            (e.g. "INFO", "DEBUG") or integer level numbers.
    loglevel_file
            File log level for Loguru sinks. Supports Loguru level names
            or integer level numbers.
    include_user_logs
            If True, include non-swarmrl Loguru records (for user/application logs)
            in the configured sinks.
    remove_default_sink
            If True, remove Loguru's default stderr sink (id 0). This avoids duplicated
            output and default DEBUG-level spam when custom sinks are configured.
    log_jax_values
            If False, disable JAX runtime value logging callbacks produced via
            ``log_jax_runtime_value`` to prevent very verbose output.

    """
    if remove_default_sink:
        try:
            logger.remove(0)
        except ValueError:
            pass

    for sink_id in _LOGGING_CONFIG.sink_ids:
        try:
            logger.remove(sink_id)
        except ValueError:
            pass
    _LOGGING_CONFIG.sink_ids = []

    loglevel_terminal = (
        loglevel_terminal.upper()
        if isinstance(loglevel_terminal, str)
        else int(loglevel_terminal)
    )
    loglevel_file = (
        loglevel_file.upper() if isinstance(loglevel_file, str) else int(loglevel_file)
    )

    log_format = "[<level>{level: <10}</level>] {time:YYYY-MM-DD HH:mm:ss}: {message}"

    def _sink_filter(record: dict) -> bool:
        if include_user_logs:
            return True
        return record["name"].startswith("swarmrl")

    active_level_numbers = [_to_level_number(loglevel_terminal)]

    if filename:
        _LOGGING_CONFIG.sink_ids.append(
            logger.add(
                filename,
                level=loglevel_file,
                format=log_format,
                filter=_sink_filter,
            )
        )
        active_level_numbers.append(_to_level_number(loglevel_file))

    _LOGGING_CONFIG.sink_ids.append(
        logger.add(
            sys.stderr,
            level=loglevel_terminal,
            format=log_format,
            filter=_sink_filter,
        )
    )

    set_jax_runtime_log_enabled(log_jax_values)
    set_jax_runtime_log_level(min(active_level_numbers))
    logger.enable("swarmrl")


def log_jax_runtime_value(
    label: str, value, level: typing.Union[str, int] = "DEBUG"
) -> None:
    """Log JAX runtime values with the global Loguru logger.

    The callback is only registered when the requested log level is enabled.
    """

    # JAX captures this branch during tracing. If the logger level changes later,
    # previously compiled paths may keep the old behavior until retracing happens.
    if not _LOGGING_CONFIG.jax_runtime_log_enabled:
        return

    level_no = _to_level_number(level)

    if level_no < _LOGGING_CONFIG.jax_runtime_log_level_no:
        return

    def _emit(x):
        logger.log(level, "{label} = {value}", label=label, value=x)

    jax.debug.callback(_emit, value, ordered=True)
