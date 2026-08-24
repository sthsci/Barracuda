"""Access genuine per-chain PyMC SMC stage and tempering progress."""

try:
    from ._backends.event_counts.smc_progress import (
        SMCProgressCallback,
        run_with_smc_progress,
    )
except ModuleNotFoundError as exc:
    if not exc.name or not exc.name.startswith("barracuda._backends"):
        raise
    from section_1.src.smc_progress import (
        SMCProgressCallback,
        run_with_smc_progress,
    )

__all__ = ["SMCProgressCallback", "run_with_smc_progress"]
