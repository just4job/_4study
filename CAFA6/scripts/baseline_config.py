"""Hyperparameters aligned with Struct2GO baseline (Table 1), per GO branch."""

# Paper default is 0.3; MF/BP train poorly at 0.3 with PPI+BCE (see log mf 2026-05-24).
BRANCH_BASELINE_DROPOUT: dict[str, float] = {
    "mf": 0.1,
    "cc": 0.2,
    "bp": 0.1,
}

BASELINE_EVAL_THRESH: dict[str, float] = {
    "mf": 0.71,
    "cc": 0.5,
    "bp": 0.4,
}

BASELINE_BATCH_SIZE = 64
BASELINE_LR = 1e-4


def baseline_checkpoint_name(branch: str, dropout: float | None = None) -> str:
    dr = dropout if dropout is not None else BRANCH_BASELINE_DROPOUT[branch]
    dr_s = f"{dr:g}"
    return f"bestmodel_{branch}_{BASELINE_BATCH_SIZE}_0.0001_{dr_s}.pkl"
