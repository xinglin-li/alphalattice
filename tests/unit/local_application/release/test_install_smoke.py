"""The install smoke launches the wheel it built with the documented command's flags."""

import itertools
import zipfile
from pathlib import Path

import pytest
from scripts.release import install_smoke

TEST_SOURCE = "scripts/release/install_smoke.py"


@pytest.mark.proves("local_application.the_install_smoke_launches_its_build_on_its_python")
@pytest.mark.parametrize(
    ("requires", "python"),
    [
        pytest.param(">=3.14,<3.15", "3.14", id="this-tree"),
        pytest.param(">=3.12,<3.13", "3.12", id="an-earlier-release"),
    ],
)
def test_the_smoke_launches_its_wheel_on_the_python_that_wheel_declares(
    tmp_path: Path, requires: str, python: str
) -> None:
    """The documented command installs the published release, so the smoke names its own wheel's
    interpreter in its place, never the published one's."""
    wheel = tmp_path / "alphalattice-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as package:
        package.writestr(
            "alphalattice-0.0.0.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: alphalattice\nRequires-Python: {requires}\n",
        )
    command = [  # the documented command's form, for the published release
        *("uvx", "--isolated", "--python", "3.12", "--from", "published.whl"),
        *("--with-requirements", "published-req.txt", "--overrides", "published-over.txt"),
        *("alphalattice", "--workspace", "./alphalattice-workspace", "serve", "--no-browser"),
    ]
    prefix = install_smoke.wheel_launcher(command, wheel, "built.whl", "req.txt", "over.txt")
    flags = dict(itertools.pairwise(prefix))
    assert flags["--python"] == python
    assert (flags["--from"], flags["--with-requirements"], flags["--overrides"]) == (
        "built.whl",
        "req.txt",
        "over.txt",
    )
