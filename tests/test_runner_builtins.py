"""
Unit tests for cyclic.runner.create_safe_globals().

These tests exercise create_safe_globals() directly (no Docker required).
Importing cyclic.runner does not register the audit hook; that only happens
inside main().
"""

from cyclic.runner import create_safe_globals


def test_class_definitions_and_isinstance_work():
    """Class statements (which need __build_class__) and isinstance/super
    must work under the restricted globals, since the allowlist that used to
    omit __build_class__ broke every `class` statement."""
    safe_globals = create_safe_globals()

    code = """
class Animal:
    def speak(self):
        return "..."

class Dog(Animal):
    def speak(self):
        return super().speak() + "Woof"

d = Dog()
assert isinstance(d, Animal)
result = d.speak()
"""
    exec(code, safe_globals)
    assert safe_globals["result"] == "...Woof"


def test_denied_builtins_are_absent():
    """eval, exec, compile, input, and breakpoint must be stripped from the
    restricted builtins dict."""
    safe_globals = create_safe_globals()
    restricted_builtins = safe_globals["__builtins__"]

    for name in ("eval", "exec", "compile", "input", "breakpoint"):
        assert name not in restricted_builtins
