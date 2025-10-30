"""Dataclasses and type definitions for profile memory extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...types import Memory, MemoryType
from ..base_memory_extractor import MemoryExtractRequest


@dataclass
class ProjectInfo:
    """Project participation information."""

    project_id: str
    project_name: str
    entry_date: str
    subtasks: Optional[List[Dict[str, Any]]] = None
    user_objective: Optional[List[Dict[str, Any]]] = None
    contributions: Optional[List[Dict[str, Any]]] = None
    user_concerns: Optional[List[Dict[str, Any]]] = None


@dataclass
class ImportanceEvidence:
    """Aggregated evidence indicating user importance within a group."""

    user_id: str
    group_id: str
    speak_count: int = 0
    refer_count: int = 0
    conversation_count: int = 0


@dataclass
class GroupImportanceEvidence:
    """Group-level importance assessment for a user."""

    group_id: str
    evidence_list: List[ImportanceEvidence]
    is_important: bool


@dataclass
class ProfileMemory(Memory):
    """
    Profile memory result class.

    Contains user profile information extracted from conversations.
    All list attributes now contain dicts with 'value' and 'evidences' fields.
    """

    user_name: Optional[str] = None

    # Skills: [{"value": "Python", "level": "高级", "evidences": ["2024-01-01|conv_123"]}]
    # Legacy format: [{"skill": "Python", "level": "高级", "evidences": ["..."]}]
    hard_skills: Optional[List[Dict[str, Any]]] = None
    soft_skills: Optional[List[Dict[str, Any]]] = None

    output_reasoning: Optional[str] = None
    

    # Other attributes: [{"value": "xxx", "evidences": ["2024-01-01|conv_123"]}]
    way_of_decision_making: Optional[List[Dict[str, Any]]] = None
    personality: Optional[List[Dict[str, Any]]] = None
    projects_participated: Optional[List[ProjectInfo]] = None
    user_goal: Optional[List[Dict[str, Any]]] = None
    work_responsibility: Optional[List[Dict[str, Any]]] = None
    working_habit_preference: Optional[List[Dict[str, Any]]] = None
    interests: Optional[List[Dict[str, Any]]] = None
    tendency: Optional[List[Dict[str, Any]]] = None
    
    # Motivational attributes: [{"value": "achievement", "level": "high", "evidences": ["2024-01-01|conv_123"]}]
    motivation_system: Optional[List[Dict[str, Any]]] = None
    fear_system: Optional[List[Dict[str, Any]]] = None
    value_system: Optional[List[Dict[str, Any]]] = None
    humor_use: Optional[List[Dict[str, Any]]] = None
    colloquialism: Optional[List[Dict[str, Any]]] = None

    group_importance_evidence: Optional[GroupImportanceEvidence] = None

    def __post_init__(self) -> None:
        """Ensure the memory type is set to PROFILE."""
        self.memory_type = MemoryType.PROFILE
        super().__post_init__()


@dataclass
class ProfileMemoryExtractRequest(MemoryExtractRequest):
    """Request payload used by ProfileMemoryExtractor."""

    pass
