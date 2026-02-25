"""Test ResearchPlan model and structured plan generation."""
import pytest
from app.models.research_plan import ResearchPlan, ResearchStep, StepStatus


class TestResearchPlanModel:
    """Tests for the ResearchPlan model."""

    def test_research_step_creation(self):
        """Test creating a ResearchStep."""
        step = ResearchStep(
            id="step-1",
            query="Who won Best Holy Local Band in Phoenix New Times 2006?",
            purpose="Identify the band that won the award",
            dependencies=[],
            status=StepStatus.PENDING,
        )
        assert step.id == "step-1"
        assert step.status == StepStatus.PENDING
        assert step.dependencies == []

    def test_research_step_with_dependencies(self):
        """Test creating a step with dependencies."""
        step = ResearchStep(
            id="step-2",
            query="What songs by the artist went viral on TikTok?",
            purpose="Find viral songs",
            dependencies=["step-1"],
            status=StepStatus.PENDING,
        )
        assert step.dependencies == ["step-1"]

    def test_research_step_revised(self):
        """Test creating a revised step."""
        step = ResearchStep(
            id="step-2-revised",
            query="Updated query for better results",
            purpose="Improved purpose",
            dependencies=["step-1"],
            status=StepStatus.PENDING,
            revised_from="step-2",
            revision_reason="Original step returned poor results",
        )
        assert step.revised_from == "step-2"
        assert step.revision_reason == "Original step returned poor results"

    def test_research_plan_creation(self):
        """Test creating a ResearchPlan."""
        steps = [
            ResearchStep(
                id="step-1",
                query="Who won Best Holy Local Band in 2006?",
                purpose="Identify the winning band",
                dependencies=[],
            ),
            ResearchStep(
                id="step-2",
                query="Who was the drummer in that band?",
                purpose="Find the drummer",
                dependencies=["step-1"],
            ),
        ]
        plan = ResearchPlan(
            id="plan-001",
            original_query="Who was the drummer in the band that won Best Holy Local Band in 2006?",
            steps=steps,
            version=1,
            created_at="2026-02-26T00:00:00Z",
            updated_at="2026-02-26T00:00:00Z",
        )
        assert plan.id == "plan-001"
        assert len(plan.steps) == 2
        assert plan.version == 1

    def test_research_plan_version_increment(self):
        """Test that version increments correctly."""
        steps = [
            ResearchStep(
                id="step-1",
                query="Test query",
                purpose="Test purpose",
                dependencies=[],
            ),
        ]
        original_plan = ResearchPlan(
            id="plan-001",
            original_query="Test query",
            steps=steps,
            version=1,
            created_at="2026-02-26T00:00:00Z",
            updated_at="2026-02-26T00:00:00Z",
        )
        # Simulate adding revised steps
        revised_steps = steps + [
            ResearchStep(
                id="step-2",
                query="Revised query",
                purpose="Revised purpose",
                dependencies=["step-1"],
                status=StepStatus.PENDING,
                revised_from="step-1",
            )
        ]
        revised_plan = ResearchPlan(
            id=original_plan.id,
            original_query=original_plan.original_query,
            steps=revised_steps,
            version=original_plan.version + 1,
            created_at=original_plan.created_at,
            updated_at="2026-02-26T01:00:00Z",
        )
        assert revised_plan.version == 2

    def test_step_status_values(self):
        """Test all StepStatus enum values."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.REVISED.value == "revised"
