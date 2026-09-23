"""Shell 执行结果。"""

from pydantic import BaseModel, Field


class ShellExecutionResult(BaseModel):
    returncode: int
    stdout: str = ""
    stderr: str = ""
    created_files: list[str] = Field(default_factory=list)

