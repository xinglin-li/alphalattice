"""The process config port holds and serves its admitted environment snapshot."""

import logging

import pytest

from alphalattice.kernel.config import Change, Config, current, declare, override, using

TEST_SOURCE = "src/alphalattice/kernel/config/port.py"


@pytest.mark.proves("config.environment_access_uses_the_config_port")
@pytest.mark.parametrize(
    "specimen", ("port-read", "secret-log", "test-override", "declared-change")
)
def test_the_environment_is_held_and_read_through_its_config_port(
    specimen: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The port keeps one snapshot and changes it only through a declaration."""
    credential = "planted-secret-for-config-specimen"
    supplied = {
        "OMP_NUM_THREADS": "4",
        "ALPHALATTICE_RETENTION_DAYS": "7",
        "DATABASE_URL": credential,
    }
    reads = 0

    def read():
        nonlocal reads
        reads += 1
        return supplied

    config = Config.capture(read)
    supplied["OMP_NUM_THREADS"] = "11"
    with using(config):
        match specimen:
            case "port-read":
                assert current().get("OMP_NUM_THREADS") == "4"
                policies = current().settings
                assert (policies.cpu_cores, policies.retention_days) == (4, 7)
                assert (
                    policies.process_memory_mode == "enforce"
                )  # unset: each declared tree is held
            case "secret-log":
                logging.getLogger(__name__).warning(
                    "database=%s child=%s",
                    current().secret("DATABASE_URL"),
                    current().child_environment(),
                )
                assert caplog.messages == ["database=<redacted> child=<private child environment>"]
                assert credential not in caplog.text and "DATABASE_URL" not in current().public
            case "test-override":
                with override(Change("test-override", {"OMP_NUM_THREADS": "2"})):
                    assert current().settings.cpu_cores == 2
                assert current().settings.cpu_cores == 4
            case "declared-change":
                changed = declare(Change("operator-policy", {"ALPHALATTICE_NETWORK_DISABLED": "1"}))
                assert current().settings.network_disabled and not config.settings.network_disabled
                assert changed.event.kind == "operator-policy"
                assert changed.event.names == ("ALPHALATTICE_NETWORK_DISABLED",)
    assert reads == 1
