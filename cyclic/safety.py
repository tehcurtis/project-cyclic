import ast
from dataclasses import dataclass
from typing import List, Set

@dataclass
class SafetyReport:
    is_safe: bool
    issues: List[str]

class SafetyScanner(ast.NodeVisitor):
    def __init__(self):
        self.issues: List[str] = []
        # Blocked modules
        self.blocked_imports: Set[str] = {'os', 'subprocess', 'sys', 'socket', 'shutil', 'importlib'}
        # Blocked built-in functions
        self.blocked_functions: Set[str] = {'eval', 'exec', 'open', '__import__'}

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name in self.blocked_imports:
                self.issues.append(f"Importing '{alias.name}' is not allowed.")
            # Check for submodules like 'os.path' if 'os' is banned
            base_module = alias.name.split('.')[0]
            if base_module in self.blocked_imports:
                self.issues.append(f"Importing '{alias.name}' (submodule of '{base_module}') is not allowed.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            module_name = node.module
            if module_name in self.blocked_imports:
                self.issues.append(f"Importing from '{module_name}' is not allowed.")
            base_module = module_name.split('.')[0]
            if base_module in self.blocked_imports:
                 self.issues.append(f"Importing from '{module_name}' (submodule of '{base_module}') is not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.blocked_functions:
                self.issues.append(f"Calling '{node.func.id}' is not allowed.")
        self.generic_visit(node)

def scan_code(code: str) -> SafetyReport:
    """
    Parses the code and runs the SafetyScanner visitor.
    Returns a SafetyReport indicating if the code is safe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return SafetyReport(is_safe=False, issues=[f"SyntaxError: {e}"])

    scanner = SafetyScanner()
    scanner.visit(tree)

    return SafetyReport(
        is_safe=len(scanner.issues) == 0,
        issues=scanner.issues
    )
