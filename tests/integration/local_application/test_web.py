"""Native OTLP uses the existing local HTTP admission and decoder."""

import json

import pytest

from alphalattice.interface.local_application import web

TEST_SOURCE = "src/alphalattice/interface/local_application/web.py"


@pytest.mark.proves("local_application.agent_otlp_keeps_host_authorization")
def test_native_otlp_uses_the_same_loopback_authorization_before_decoding():
    """Binary is admitted only for the named OTLP route, with every existing secret checked."""
    application = web.LocalWebApplication(external_workspace_id="workspace-1")
    application.allowed_hosts = frozenset({"127.0.0.1:12345"})
    received = []

    @application.route(
        "POST", "/v1/logs", mutates=True, external_client=True, otlp=("logs", "catalog-1")
    )
    def receive(_query, payload):
        received.extend(payload["rows"])
        return web.LocalWebResponse(web.HTTPStatus.OK, b"", "application/x-protobuf")

    headers = {
        "Host": "127.0.0.1:12345",
        web.WORKSPACE_HEADER: "workspace-1",
        web.INSTANCE_HEADER: application.external_instance,
        web.CLIENT_HEADER: application.external_token,
        "Content-Type": "application/x-protobuf",
        "Content-Length": "0",
    }
    for change in (
        {"Host": "example.com:12345"},
        {web.CLIENT_HEADER: "wrong"},
        {"Origin": "http://127.0.0.1:12345"},
    ):
        refused = application.preflight(method="POST", target="/v1/logs", headers=headers | change)
        assert isinstance(refused, web.LocalWebResponse) and refused.status == 403
        assert received == []
    admitted = application.preflight(method="POST", target="/v1/logs", headers=headers)
    assert isinstance(admitted, web.AdmittedRequest)
    assert application.dispatch(admitted, b"").status == 200
    payload = json.dumps(
        {
            "resourceLogs": [
                {
                    "scopeLogs": [
                        {
                            "logRecords": [
                                {
                                    "body": {"stringValue": "PRIVATE_BODY"},
                                    "attributes": [
                                        {"key": "session.id", "value": {"stringValue": "session-1"}}
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ).encode()
    admitted = application.preflight(
        method="POST",
        target="/v1/logs",
        headers=headers
        | {
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )
    assert isinstance(admitted, web.AdmittedRequest)
    assert application.dispatch(admitted, payload).status == 200
    assert received[0]["session_id"] == "session-1" and "PRIVATE_BODY" not in json.dumps(received)
    other = web.LocalWebApplication(external_workspace_id="workspace-2")
    other.allowed_hosts = application.allowed_hosts
    other.route("POST", "/v1/logs", mutates=True, external_client=True, otlp=("logs", "catalog-2"))(
        receive
    )
    admitted = other.preflight(
        method="POST",
        target="/v1/logs",
        headers=headers
        | {
            web.WORKSPACE_HEADER: "workspace-2",
            web.INSTANCE_HEADER: other.external_instance,
            web.CLIENT_HEADER: other.external_token,
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )
    assert isinstance(admitted, web.AdmittedRequest)
    assert other.dispatch(admitted, payload).status == 200
    assert [row["workspace_id"] for row in received] == ["catalog-1", "catalog-2"]
    assert received[0]["event_id"] != received[1]["event_id"]
