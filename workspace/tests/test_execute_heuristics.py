"""Static unit tests for the execute-stage heuristics.

`is_read_only` is the load-bearing classifier for the anti-paralysis guard.
If it misjudges command intent, the guard either fires too early (aborting
real progress) or too late (letting infinite loops run). Pin the behavior.
"""

from __future__ import annotations

from src.stages.execute import is_read_only


def test_ls_is_read_only():
    assert is_read_only("ls -la")


def test_grep_is_read_only():
    assert is_read_only("grep -rn 'foo' .")


def test_find_is_read_only():
    assert is_read_only("find . -name '*.py'")


def test_redirection_is_write():
    assert not is_read_only("echo hello > out.txt")


def test_append_redirection_is_write():
    assert not is_read_only("echo hello >> out.txt")


def test_heredoc_is_write():
    assert not is_read_only("cat > foo.py <<'EOF'\nprint('hi')\nEOF")


def test_pip_install_is_write():
    assert not is_read_only("pip install requests")


def test_sed_inplace_is_write():
    assert not is_read_only("sed -i 's/foo/bar/' file.txt")


def test_python_dash_c_is_write():
    assert not is_read_only("python -c 'open(\"x\",\"w\").write(1)'")


def test_pwd_is_read_only():
    assert is_read_only("pwd")
