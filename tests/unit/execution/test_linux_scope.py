import errno
from pathlib import Path

import pytest

from alphalattice.kernel.execution.linux_scope import LinuxScope

TEST_SOURCE = "src/alphalattice/kernel/execution/linux_scope.py"


@pytest.mark.proves("execution.a_removed_cgroup_reads_as_stopped")
@pytest.mark.parametrize(
    ("code", "ended"),
    [(errno.ENODEV, True), (errno.ENOENT, True), (errno.EACCES, False)],
    ids=("removed-after-kill", "gone-before-read", "unreadable"),
)
def test_a_cgroup_its_manager_removed_after_the_kill_reads_as_stopped(
    code, ended, tmp_path, monkeypatch
):
    """A killed cgroup whose emptied scope its manager removed (ENODEV, or the file gone) has ended;
    any other refusal to read it is raised, never read as stopped."""
    (tmp_path / "cgroup.kill").write_bytes(b"0")
    opened = Path.open

    def removed(path, *args, **kwargs):
        if path.name == "cgroup.events":
            raise OSError(code, "the cgroup's events file")
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", removed)
    scope = LinuxScope(tmp_path, subgroup=True)
    if ended:
        scope.stop()
    else:
        with pytest.raises(PermissionError):
            scope.stop()
    assert (tmp_path / "cgroup.kill").read_bytes() == b"1"
