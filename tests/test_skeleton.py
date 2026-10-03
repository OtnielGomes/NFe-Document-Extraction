import importlib

PROJECT_MODULES = (
    "nfe_document_extraction.domain",
    "nfe_document_extraction.validation",
    "nfe_document_extraction.agents",
    "nfe_document_extraction.graph",
    "nfe_document_extraction.evaluation",
    "nfe_document_extraction.app",
)


def test_project_modules_import() -> None:
    for module_name in PROJECT_MODULES:
        importlib.import_module(module_name)
