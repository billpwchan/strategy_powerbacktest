"""Exception types raised by the package."""


class PowerBacktestError(Exception):
    """Base class for all errors raised by powerbacktest."""


class ConfigError(PowerBacktestError):
    """Invalid or inconsistent configuration."""


class DataError(PowerBacktestError):
    """Market data is missing, malformed or could not be fetched."""


class DataSourceError(DataError):
    """The upstream data source (Futu OpenD) returned an error."""


class QuotaExceededError(DataSourceError):
    """Fetching a new symbol would exceed the Futu historical K-line quota."""


class StrategyError(PowerBacktestError):
    """A strategy is unknown, misconfigured or produced invalid output."""
