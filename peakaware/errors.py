class PeakAwareError(RuntimeError):
    """Base error raised by PeakAware."""


class CaptureError(PeakAwareError):
    pass


class UnsupportedTorchVersionError(PeakAwareError):
    pass


class UnsupportedGraphError(PeakAwareError):
    pass


class InfeasibleBudgetError(PeakAwareError):
    pass


class CostUnavailableError(PeakAwareError):
    pass


class PartitionError(PeakAwareError):
    pass


class PartitionABIError(PartitionError):
    pass


class PlanValidationError(PeakAwareError):
    pass


class PluginConflictError(PeakAwareError):
    pass


class PatchRestoreError(PeakAwareError):
    pass
