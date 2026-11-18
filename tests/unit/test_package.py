def test_package_exposes_a_version() -> None:
    from ragforge import __version__

    assert __version__ == "0.1.0"
