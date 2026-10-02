"""Explicit private dependency injection into immutable parent function bodies.

No parent globals, module registrations or code objects are modified. Each
binding gets its own globals dictionary, shared by its local function clones.
"""
from types import FunctionType


def bind(module, overrides=None, package=None):
    namespace = dict(vars(module))
    if package is not None:
        namespace['__package__'] = package
    for name, value in vars(module).items():
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            clone = FunctionType(value.__code__, namespace, name,
                                 value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            clone.__doc__ = value.__doc__
            namespace[name] = clone
    namespace.update(overrides or {})
    return namespace
