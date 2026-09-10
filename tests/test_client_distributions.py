import importlib.metadata
import warnings

import pytest

import openeo


def _fake_version_available(names):
    def _version(name):
        if name in names:
            return "1.2.3"
        raise importlib.metadata.PackageNotFoundError(name)

    return _version


def test_check_distribution_conflict_warns_when_both_installed(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        _fake_version_available({"openeo", "openeo-python-client-dedl"}),
    )
    with pytest.warns(RuntimeWarning, match="Multiple openEO client distributions"):
        openeo._check_distribution_conflict()


def test_check_distribution_conflict_silent_with_single_distribution(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", _fake_version_available({"openeo"}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        openeo._check_distribution_conflict()
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_client_version_prefers_dedl_distribution(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", _fake_version_available({"openeo", "openeo-python-client-dedl"}))
    assert openeo.client_version() == "1.2.3"


def test_client_version_falls_back_to_module_version(monkeypatch):
    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert openeo.client_version() == openeo._version.__version__
