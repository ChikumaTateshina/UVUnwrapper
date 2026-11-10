"""型付きエラーと検証レポート。

エラーはコード、対象(Object/面/コーナー)、失敗段階、変更有無、
ロールバック状態、推奨対処を持つ (設計仕様 10章)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class ErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    CONSTRAINT_CONFLICT = "CONSTRAINT_CONFLICT"
    NO_FEASIBLE_LAYOUT = "NO_FEASIBLE_LAYOUT"
    UNWRAP_FAILED = "UNWRAP_FAILED"
    STALE_INPUT = "STALE_INPUT"
    COMMIT_FAILED = "COMMIT_FAILED"
    RESTORE_CONFLICT = "RESTORE_CONFLICT"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    SHARED_MESH = "SHARED_MESH"
    NOT_EDITABLE = "NOT_EDITABLE"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    NO_OP = "NO_OP"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Target:
    """問題箇所。face/corner/edge は該当しなければ None。"""
    object_key: str
    face: Optional[int] = None
    corner: Optional[int] = None
    edge: Optional[Tuple[int, int]] = None

    def describe(self) -> str:
        parts = [self.object_key]
        if self.face is not None:
            parts.append(f"face={self.face}")
        if self.corner is not None:
            parts.append(f"corner={self.corner}")
        if self.edge is not None:
            parts.append(f"edge={self.edge}")
        return " ".join(parts)


@dataclass
class Issue:
    code: ErrorCode
    severity: Severity
    message: str
    stage: str = ""
    targets: List[Target] = field(default_factory=list)
    advice: str = ""

    def describe(self, max_targets: int = 5) -> str:
        text = f"[{self.code.value}] {self.message}"
        if self.stage:
            text += f" (段階: {self.stage})"
        if self.targets:
            shown = ", ".join(t.describe() for t in self.targets[:max_targets])
            more = len(self.targets) - max_targets
            text += f" 対象: {shown}" + (f" 他{more}件" if more > 0 else "")
        if self.advice:
            text += f" 対処: {self.advice}"
        return text


@dataclass
class ValidationReport:
    issues: List[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def add(self, code: ErrorCode, severity: Severity, message: str, *,
            stage: str = "", targets=None, advice: str = "") -> Issue:
        issue = Issue(code, severity, message, stage, list(targets or []), advice)
        self.issues.append(issue)
        return issue

    def error(self, code: ErrorCode, message: str, **kw) -> Issue:
        return self.add(code, Severity.ERROR, message, **kw)

    def warn(self, code: ErrorCode, message: str, **kw) -> Issue:
        return self.add(code, Severity.WARNING, message, **kw)

    def info(self, code: ErrorCode, message: str, **kw) -> Issue:
        return self.add(code, Severity.INFO, message, **kw)

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)

    def raise_if_failed(self, stage: str = "") -> None:
        if not self.ok:
            first = self.errors[0]
            raise SUOError(first.code, first.message, stage=stage or first.stage,
                           targets=first.targets, advice=first.advice, report=self)


class SUOError(Exception):
    """アドオン内の全ての想定済み失敗。握りつぶして成功扱いしない。"""

    def __init__(self, code: ErrorCode, message: str, *, stage: str = "",
                 targets=None, advice: str = "", changed: bool = False,
                 rollback_state: str = "NOT_NEEDED",
                 report: Optional[ValidationReport] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.targets = list(targets or [])
        self.advice = advice
        self.changed = changed
        self.rollback_state = rollback_state
        self.report = report

    def describe(self) -> str:
        issue = Issue(self.code, Severity.ERROR, self.message, self.stage,
                      self.targets, self.advice)
        text = issue.describe()
        text += f" 変更: {'あり' if self.changed else 'なし'} / ロールバック: {self.rollback_state}"
        return text

    def __str__(self) -> str:
        return self.describe()
