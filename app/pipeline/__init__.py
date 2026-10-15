"""
Pipeline package for DEVSCOUT Learning Companion.

Exposes the 9-stage pipeline modules along with semantic aliases:
1 Developer Goal & Skill Context
2 Intent & Learning Plan
3 Research & Source Discovery
4 Evidence Review
5 Explanation & Concept Mapping
6 Guided Project Work
7 Understanding Check
8 Code Assistance Gate
9 Progress & Reflection
"""
from app.pipeline import (
    stage1_developer_input,
    stage2_input_processing,
    stage3_research,
    stage4_evidence_validation,
    stage5_ai_analysis,
    stage6_recommendation,
    stage7_output_validation,
    stage8_decision_report,
    stage9_developer_feedback,
    response_validator,
)

# Semantic stage aliases
stage1_goal_context = stage1_developer_input
stage2_learning_plan = stage2_input_processing
stage3_source_discovery = stage3_research
stage4_evidence_review = stage4_evidence_validation
stage5_concept_mapping = stage5_ai_analysis
stage6_guided_work = stage6_recommendation
stage7_understanding_check = stage7_output_validation
stage8_code_gate = stage8_decision_report
stage9_progress_reflection = stage9_developer_feedback

# Cross-cutting validator exports
validate_response = response_validator.validate_response
repair_response = response_validator.repair_response
ValidationReport = response_validator.ValidationReport

__all__ = [
    # Original file module names
    "stage1_developer_input",
    "stage2_input_processing",
    "stage3_research",
    "stage4_evidence_validation",
    "stage5_ai_analysis",
    "stage6_recommendation",
    "stage7_output_validation",
    "stage8_decision_report",
    "stage9_developer_feedback",
    # Semantic stage aliases
    "stage1_goal_context",
    "stage2_learning_plan",
    "stage3_source_discovery",
    "stage4_evidence_review",
    "stage5_concept_mapping",
    "stage6_guided_work",
    "stage7_understanding_check",
    "stage8_code_gate",
    "stage9_progress_reflection",
    # Validator
    "response_validator",
    "validate_response",
    "repair_response",
    "ValidationReport",
]
