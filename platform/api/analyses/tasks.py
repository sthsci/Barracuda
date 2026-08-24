from __future__ import annotations

from celery import shared_task

from .execution import execute_analysis_job


@shared_task(name="analyses.execute_analysis_job", ignore_result=True)
def run_analysis_job(job_id: str):
    execute_analysis_job(job_id)


def dispatch_analysis_job(job) -> str:
    result = run_analysis_job.apply_async(args=(str(job.pk),))
    task_id = str(result.id or "")
    if task_id:
        job.__class__.objects.filter(pk=job.pk).update(task_id=task_id)
        job.task_id = task_id
    return task_id
