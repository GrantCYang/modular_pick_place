# execution/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class ObjectResult:
    """单个物体的执行结果"""
    object_id: str
    success: bool
    failure_reason: str = ""    # 失败时的原因描述，如 "ik_failed"、"grasp_slip"
    steps_taken: int = 0


@dataclass
class ExecutionResult:
    """
    执行模块的输出，汇总本次 episode 所有物体的执行情况。
    """
    object_results: List[ObjectResult] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.object_results if r.success)

    @property
    def n_total(self) -> int:
        return len(self.object_results)

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_total if self.n_total > 0 else 0.0

    def summary(self) -> str:
        lines = [f"[ExecutionResult] {self.n_success}/{self.n_total} objects succeeded"]
        for r in self.object_results:
            status = "✓" if r.success else f"✗ ({r.failure_reason})"
            lines.append(f"  {r.object_id}: {status}")
        return "\n".join(lines)


class BaseExecutor(ABC):
    @abstractmethod
    def execute(self, action_sequence: "ActionSequence") -> ExecutionResult:
        """
        Args:
            action_sequence: 来自规划模块的有序动作列表
        Returns:
            ExecutionResult: 每个物体的执行成败及原因
        """
        ...

    def reset(self) -> None:
        """episode 开始时重置执行器内部状态。默认无操作。"""
        pass