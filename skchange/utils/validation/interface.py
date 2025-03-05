"""Decorators for enforcing interface implementation."""


def overrides(interface_class):
    """Indicate that an attribute/method implements interface."""

    def overrider(method):
        assert method.__name__ in dir(
            interface_class
        ), f"'{method.__name__}' not in {interface_class}."
        return method

    return overrider
