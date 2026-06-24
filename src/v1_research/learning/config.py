"""Learning-rule factory configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from v1_research.learning.bcm import BCMLearningRule, BCMConfig
from v1_research.learning.rules import LearningRule

LearningKind = Literal["bcm"]


@dataclass(frozen=True, slots=True)
class LearningConfig:
    """Top-level learning-rule selector."""

    kind: LearningKind = "bcm"
    bcm: BCMConfig = field(default_factory=BCMConfig)


def make_learning_rule(cfg: LearningConfig) -> LearningRule:
    """Creates the selected learning rule."""

    if cfg.kind == "bcm":
        return BCMLearningRule(cfg.bcm)
    raise ValueError(f"Unknown learning rule kind: {cfg.kind!r}")
