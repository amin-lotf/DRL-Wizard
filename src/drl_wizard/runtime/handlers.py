from drl_wizard.backend.services.training_service.jobs import JobState
from drl_wizard.runtime.event_registery import set_stop_event


def stop_event_handler(job:JobState):
    if job.stop_requested:
        set_stop_event(job.job_id)