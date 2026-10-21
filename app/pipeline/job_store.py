"""
app/pipeline/job_store.py

In-memory job tracking store for asynchronous learning pipeline runs.
Tracks job status, per-stage progress, results, and errors.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from app.models.pipeline import JobStatus, LearnJobRecord, LearningResponse


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JobStore:
    """In-memory store for tracking pipeline job states."""

    def __init__(self) -> None:
        self._jobs: dict[str, LearnJobRecord] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, job_id: str | None = None) -> LearnJobRecord:
        """Create a new job record with PENDING status."""
        async with self._lock:
            record = LearnJobRecord(
                status=JobStatus.PENDING,
                progress=0.0,
                current_stage="initializing",
                stage_id=0,
            )
            if job_id:
                record.job_id = job_id
            self._jobs[record.job_id] = record
            return record

    async def get_job(self, job_id: str) -> LearnJobRecord | None:
        """Retrieve a job record by job_id."""
        async with self._lock:
            return self._jobs.get(job_id)

    async def update_progress(
        self,
        job_id: str,
        stage_id: int,
        stage_name: str,
        progress: float,
    ) -> None:
        """Update job stage progress and mark as RUNNING."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.RUNNING
                job.stage_id = stage_id
                job.current_stage = stage_name
                job.progress = round(progress, 2)
                job.updated_at = _utcnow()

    async def mark_completed(self, job_id: str, result: LearningResponse) -> None:
        """Mark job as COMPLETED and attach result."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.stage_id = 9
                job.current_stage = "completed"
                job.result = result
                job.updated_at = _utcnow()

    async def mark_failed(self, job_id: str, error: str, result: LearningResponse | None = None) -> None:
        """Mark job as FAILED and record error message."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = error
                if result is not None:
                    job.result = result
                job.updated_at = _utcnow()

    async def mark_timed_out(self, job_id: str, error: str = "Pipeline execution timed out") -> None:
        """Mark job as TIMED_OUT."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.TIMED_OUT
                job.error = error
                job.updated_at = _utcnow()

    async def clear(self) -> None:
        """Clear all stored jobs (for tests)."""
        async with self._lock:
            self._jobs.clear()


_GLOBAL_JOB_STORE = JobStore()


def get_job_store() -> JobStore:
    """Return singleton JobStore instance."""
    return _GLOBAL_JOB_STORE
