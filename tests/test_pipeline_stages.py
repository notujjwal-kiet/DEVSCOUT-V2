"""
Tests for pipeline stage placeholders (stages 1-9).

Each test exercises the stage's public run() coroutine with valid stub
inputs and confirms the output type and request_id propagation.
"""
import pytest

from app.models.pipeline import (
    AnalysisResult,
    DeveloperFeedback,
    DeveloperInput,
    OutputValidationResult,
    ProcessedInput,
    RecommendationResult,
    ResearchResult,
    StageMetrics,
    ValidationResult,
)
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
)

QUESTION = "PostgreSQL vs MongoDB: which fits a multi-tenant SaaS?"


@pytest.mark.asyncio
async def test_stage1():
    result = await stage1_developer_input.run(QUESTION)
    assert isinstance(result, DeveloperInput)
    assert result.question == QUESTION


@pytest.mark.asyncio
async def test_stage2():
    s1 = await stage1_developer_input.run(QUESTION)
    result = await stage2_input_processing.run(s1)
    assert isinstance(result, ProcessedInput)
    assert result.request_id == s1.request_id


@pytest.mark.asyncio
async def test_stage3():
    s1 = await stage1_developer_input.run(QUESTION)
    s2 = await stage2_input_processing.run(s1)
    result = await stage3_research.run(s2)
    assert isinstance(result, ResearchResult)
    assert result.request_id == s1.request_id


@pytest.mark.asyncio
async def test_stage4():
    s1 = await stage1_developer_input.run(QUESTION)
    s2 = await stage2_input_processing.run(s1)
    s3 = await stage3_research.run(s2)
    result = await stage4_evidence_validation.run(s3)
    assert isinstance(result, ValidationResult)
    assert result.request_id == s1.request_id


@pytest.mark.asyncio
async def test_stage5():
    s1 = await stage1_developer_input.run(QUESTION)
    s2 = await stage2_input_processing.run(s1)
    s3 = await stage3_research.run(s2)
    s4 = await stage4_evidence_validation.run(s3)
    result = await stage5_ai_analysis.run(s4, s2.normalized_question)
    assert isinstance(result, AnalysisResult)
    assert result.request_id == s1.request_id


@pytest.mark.asyncio
async def test_stage6():
    s1 = await stage1_developer_input.run(QUESTION)
    s2 = await stage2_input_processing.run(s1)
    s3 = await stage3_research.run(s2)
    s4 = await stage4_evidence_validation.run(s3)
    s5 = await stage5_ai_analysis.run(s4, s2.normalized_question)
    result = await stage6_recommendation.run(s5)
    assert isinstance(result, RecommendationResult)
    assert len(result.recommendations) >= 1


@pytest.mark.asyncio
async def test_stage7():
    s1 = await stage1_developer_input.run(QUESTION)
    s2 = await stage2_input_processing.run(s1)
    s3 = await stage3_research.run(s2)
    s4 = await stage4_evidence_validation.run(s3)
    s5 = await stage5_ai_analysis.run(s4, s2.normalized_question)
    s6 = await stage6_recommendation.run(s5)
    result = await stage7_output_validation.run(s6, s4)
    assert isinstance(result, OutputValidationResult)


@pytest.mark.asyncio
async def test_stage8():
    s1 = await stage1_developer_input.run(QUESTION)
    s2 = await stage2_input_processing.run(s1)
    s3 = await stage3_research.run(s2)
    s4 = await stage4_evidence_validation.run(s3)
    s5 = await stage5_ai_analysis.run(s4, s2.normalized_question)
    s6 = await stage6_recommendation.run(s5)
    s7 = await stage7_output_validation.run(s6, s4)
    metrics: list[StageMetrics] = [
        m for m in [s2.metrics, s3.metrics, s4.metrics, s5.metrics,
                    s6.metrics, s7.metrics]
        if m is not None
    ]
    report = await stage8_decision_report.run(s1, s2, s3, s6, s7, metrics)
    assert report.request_id == s1.request_id
    assert report.report_id != ""


@pytest.mark.asyncio
async def test_stage9():
    feedback = DeveloperFeedback(
        report_id="report-abc",
        request_id="req-abc",
        rating=4,
        comment="Very helpful!",
    )
    ack = await stage9_developer_feedback.run(feedback)
    assert ack.feedback_id == feedback.feedback_id
