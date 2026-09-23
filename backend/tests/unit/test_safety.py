from app.auth import PermissionPolicy
from app.core.models import RiskLevel, ToolMetadata
from app.execution.sandbox import WorkspaceManager


def test_write_overwrite_requires_approval():
    policy = PermissionPolicy()
    metadata = ToolMetadata(name="vector.buffer", description="buffer", risk_level=RiskLevel.WRITE)
    decision = policy.authorize(metadata, {"overwrite": True})
    assert decision.allowed is False
    assert decision.needs_approval is True


def test_workspace_rejects_path_escape(tmp_path):
    workspace = WorkspaceManager(tmp_path / "workspace")
    try:
        workspace.resolve(tmp_path / "outside.txt")
    except PermissionError:
        pass
    else:
        raise AssertionError("path escape should be rejected")
