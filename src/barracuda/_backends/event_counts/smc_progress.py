"""Forward PyMC's native SMC progress events to BARRACUDA callers.

PyMC 5.25 runs every SMC chain in a worker process.  The workers publish
``stage`` and ``beta`` through a multiprocessing manager, and PyMC's parent
process turns those values into ``CustomProgress.update`` calls.  BARRACUDA hooks
that parent-process update, so no callback (or ``ContextVar``) has to cross a
process boundary and no progress values are estimated by the web application.
"""

from __future__ import annotations

from contextvars import ContextVar
import re
from typing import Callable


SMCProgressCallback = Callable[[int, int, float], None]

_smc_progress_callback: ContextVar[SMCProgressCallback | None] = ContextVar(
    "orca_smc_progress_callback",
    default=None,
)
_smc_status_pattern = re.compile(
    r"Stage:\s*(\d+)\s+Beta:\s*([0-9.eE+-]+)"
)
_smc_chain_pattern = re.compile(r"\bChain\s+(\d+)\b")


def _install_smc_progress_bridge() -> None:
    """Install one process-local bridge around PyMC's Rich progress class."""

    import pymc.smc.sampling as smc_sampling

    progress_class = smc_sampling.CustomProgress
    if getattr(progress_class, "_orca_progress_bridge", False):
        return

    class OrcaSMCProgress(progress_class):
        _orca_progress_bridge = True

        def add_task(self, *args, **kwargs):
            task_id = super().add_task(*args, **kwargs)
            description = (
                args[0]
                if args
                else kwargs.get("description")
            )
            if isinstance(description, str):
                match = _smc_chain_pattern.search(description)
                if match is not None:
                    chain = int(match.group(1))
                    chain_by_task = getattr(
                        self,
                        "_orca_chain_by_task",
                        None,
                    )
                    if chain_by_task is None:
                        chain_by_task = {}
                        self._orca_chain_by_task = chain_by_task
                    chain_by_task[task_id] = chain
                    status = kwargs.get("status")
                    status_match = (
                        _smc_status_pattern.search(status)
                        if isinstance(status, str)
                        else None
                    )
                    callback = _smc_progress_callback.get()
                    if callback is not None and status_match is not None:
                        callback(
                            chain,
                            int(status_match.group(1)),
                            float(status_match.group(2)),
                        )
            return task_id

        def update(self, task_id, *args, **kwargs):
            status = kwargs.get("status")
            callback = _smc_progress_callback.get()
            chain = getattr(
                self,
                "_orca_chain_by_task",
                {},
            ).get(task_id)
            if callback is not None and chain is not None and isinstance(status, str):
                match = _smc_status_pattern.search(status)
                if match is not None:
                    callback(
                        int(chain),
                        int(match.group(1)),
                        float(match.group(2)),
                    )
            return super().update(task_id, *args, **kwargs)

    smc_sampling.CustomProgress = OrcaSMCProgress


def run_with_smc_progress(callback: SMCProgressCallback | None, operation):
    """Run a synchronous PyMC operation with its native progress forwarded."""

    _install_smc_progress_bridge()
    token = _smc_progress_callback.set(callback)
    try:
        return operation()
    finally:
        _smc_progress_callback.reset(token)
