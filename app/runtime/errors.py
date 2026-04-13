class RuntimeError_(Exception):
    """Base for runtime-layer errors."""


class ModelNotFoundError(RuntimeError_):
    pass


class BackendStartError(RuntimeError_):
    pass


class BackendNotReadyError(RuntimeError_):
    pass


class InferenceError(RuntimeError_):
    pass
