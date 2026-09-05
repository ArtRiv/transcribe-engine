"""Tests for R-04 path discipline — platform/paths.py."""

import sys
from pathlib import Path

import pytest

from transcribe_engine.platform import paths


def test_bundle_root_uses_meipass_when_set(monkeypatch):
    monkeypatch.setattr(sys, "_MEIPASS", "/tmp/_MEItest", raising=False)
    assert paths.bundle_root() == Path("/tmp/_MEItest")


def test_bundle_root_falls_back_to_package_dir_when_meipass_unset(monkeypatch):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    result = paths.bundle_root()
    # Result is engine package parent; transcribe_engine/__init__.py should be reachable
    assert (result / "transcribe_engine" / "__init__.py").is_file()


@pytest.mark.skipif(sys.platform != "linux", reason="XDG_CONFIG_HOME is Linux-only convention")
def test_config_dir_honors_xdg_on_linux(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    result = paths.config_dir()
    assert str(result).startswith(str(tmp_path / "config"))


def test_cache_dir_includes_models_suffix(monkeypatch, tmp_path):
    # Force a temp root so the test does not pollute real ~/.cache
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    result = paths.cache_dir()
    assert result.name == "models"


def test_directories_created_idempotently(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    # Call twice — second call must not raise
    a = paths.config_dir()
    b = paths.config_dir()
    assert a == b
    assert a.is_dir()


def test_log_dir_distinct_from_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert paths.log_dir() != paths.config_dir()
