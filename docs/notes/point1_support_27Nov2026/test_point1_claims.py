"""pytest wrapper: the Point 1 numerical checks all PASS, and two runs print identical output."""
import contextlib
import importlib.util
import io
import pathlib

HERE = pathlib.Path(__file__).parent


def _run():
    spec = importlib.util.spec_from_file_location("p1", HERE / "point1_claims_numerics_27Nov2026.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = mod.main()
    return code, buf.getvalue()


def test_all_claims_pass_and_output_is_identical_twice():
    code1, out1 = _run()
    code2, out2 = _run()
    assert code1 == 0 and code2 == 0, out1
    assert out1 == out2
    assert "FAIL" not in out1
    assert out1.count("PASS") == 10
