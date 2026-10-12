from types import ModuleType
import sys

from .olinear import OLinear


class _CallableModule(ModuleType):
    def __call__(self, **kwargs):
        return OLinear(**kwargs)

    def required_hyper_params(self):
        if hasattr(OLinear, "required_hyper_params"):
            return OLinear.required_hyper_params()
        return {}


_current_module = sys.modules[__name__]
_current_module.__class__ = _CallableModule

__all__ = ["OLinear"]
