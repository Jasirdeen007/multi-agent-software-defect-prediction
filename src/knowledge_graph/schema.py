from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    ISSUE = "Issue"
    PROJECT = "Project"
    SOURCE = "Source"
    COMPONENT = "Component"
    SEVERITY = "Severity"
    PRIORITY = "Priority"
    STATUS = "Status"
    RESOLUTION = "Resolution"


class EdgeType(str, Enum):
    BELONGS_TO = "BELONGS_TO"          # Issue -> Project
    TRACKED_IN = "TRACKED_IN"          # Issue -> Source
    HAS_COMPONENT = "HAS_COMPONENT"    # Issue -> Component
    HAS_SEVERITY = "HAS_SEVERITY"      # Issue -> Severity
    HAS_PRIORITY = "HAS_PRIORITY"      # Issue -> Priority
    HAS_STATUS = "HAS_STATUS"          # Issue -> Status
    HAS_RESOLUTION = "HAS_RESOLUTION"  # Issue -> Resolution
