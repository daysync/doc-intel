import doc_intel


def test_package_imports_and_has_version() -> None:
    assert doc_intel.__version__ == "0.0.1"
