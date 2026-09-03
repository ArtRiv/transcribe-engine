"""Smoke tests for picker_server."""

import pytest

from transcribe_engine.picker_server import _HF_TOKEN_RE, find_free_port, save_hf_token


def test_find_free_port_returns_high_port():
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_find_free_port_does_not_collide():
    # Two consecutive calls must return different ports (OS-assigned)
    ports = {find_free_port() for _ in range(5)}
    assert len(ports) >= 2  # could be same by luck; 5 attempts = >99% diverse


def test_hf_token_regex_accepts_valid_hf_token():
    assert _HF_TOKEN_RE.match("hf_AbCdEfGhIjKlMnOpQrStUvWxYz12")


def test_hf_token_regex_rejects_short_token():
    assert _HF_TOKEN_RE.match("hf_short") is None


def test_hf_token_regex_rejects_no_prefix():
    assert _HF_TOKEN_RE.match("AbCdEfGhIjKlMnOpQrStUvWxYz12") is None


def test_save_hf_token_creates_file_with_content(tmp_path):
    path = tmp_path / "hf_token"
    save_hf_token("hf_TestTokenAaBbCcDdEe1234", path)
    assert path.read_text() == "hf_TestTokenAaBbCcDdEe1234"


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="chmod is Unix-only")
def test_save_hf_token_sets_mode_0600_on_unix(tmp_path):
    import stat
    path = tmp_path / "hf_token"
    save_hf_token("hf_TestTokenAaBbCcDdEe1234", path)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600   # T-7-03 mitigation


# D-19 — _format_time_estimate behavior
def test_format_time_estimate_vulkan_average_tier():
    from transcribe_engine.picker_server import _format_time_estimate
    # Average tier RTF=0.30 on Vulkan (mult=1.0) -> 30 * 0.30 * 1.0 = 9 min
    s = _format_time_estimate(0.30, "vulkan", "AMD Radeon RX 6600 (Vulkan)")
    assert "~9 min" in s
    assert "AMD Radeon RX 6600 (Vulkan)" in s
    assert "30-min recording" in s


def test_format_time_estimate_cpu_is_much_slower():
    from transcribe_engine.picker_server import _format_time_estimate
    v = _format_time_estimate(0.30, "vulkan", "GPU")
    c = _format_time_estimate(0.30, "cpu", "CPU")
    # CPU multiplier is 8x — string must reflect that
    v_min = int(v.split("~")[1].split(" min")[0])
    c_min = int(c.split("~")[1].split(" min")[0])
    assert c_min >= v_min * 7  # at least 7x slower (allows for rounding)


def test_format_time_estimate_metal_between_vulkan_and_cpu():
    from transcribe_engine.picker_server import _format_time_estimate
    v = int(_format_time_estimate(0.55, "vulkan", "G").split("~")[1].split(" min")[0])
    m = int(_format_time_estimate(0.55, "metal", "G").split("~")[1].split(" min")[0])
    c = int(_format_time_estimate(0.55, "cpu", "G").split("~")[1].split(" min")[0])
    assert v <= m < c


def test_format_time_estimate_unknown_backend_treats_as_cpu():
    from transcribe_engine.picker_server import _format_time_estimate
    s = _format_time_estimate(0.30, "rocm-not-yet-supported", "Foo")
    # unknown backend -> conservative (cpu multiplier)
    assert int(s.split("~")[1].split(" min")[0]) >= 30 * 0.30 * 7
