from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

import barracuda_runner

from barracuda_runner import ContractError, run_analysis, run_request, validate_configuration


class FakeAdapter:
    def __init__(self, *, artifact_path: str = "results/evidence.csv") -> None:
        self.artifact_path = artifact_path
        self.payload: bytes | None = None
        self.configuration = None

    def normalize_csv(self, payload: bytes):
        self.payload = payload
        rows = payload.decode().strip().splitlines()
        return {
            "data": {"canonical_rows": rows[1:]},
            "summary": {"rows": len(rows) - 1, "conditions": 1},
        }

    def run(self, data, *, configuration, progress, output_dir: Path):
        self.configuration = configuration
        progress(0.25)
        progress({"phase": "sampling", "fraction": 0.75, "beta": 0.5})
        path = output_dir / self.artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model,log_evidence\nhomo,-10.2\n")
        return {
            "summary": {
                "best_model": "homo",
                "canonical_rows": len(data["canonical_rows"]),
            },
            "artifacts": [
                {
                    "role": "model-evidence",
                    "path": self.artifact_path,
                    "media_type": "text/csv",
                }
            ],
        }


class FakeRuntime:
    def __init__(self, adapter: FakeAdapter) -> None:
        self.adapter = adapter
        self.requested: list[str] = []

    def get_adapter(self, analysis_type: str):
        self.requested.append(analysis_type)
        return self.adapter


def test_count_configuration_is_strict_and_materializes_safe_defaults() -> None:
    request = validate_configuration(
        "event_count_donor_ignorant",
        {
            "schema_version": 1,
            "models": ["homo", "hetero3"],
            "sampler": {"particles": 64, "chains": 2, "cores": 1},
        },
    )

    assert request.configuration["observation_time"] == 1.0
    assert request.configuration["models"] == ["homo", "hetero3"]
    assert request.configuration["sampler"] == {
        "particles": 64,
        "chains": 2,
        "cores": 1,
        "seed": None,
        "threshold": 0.5,
        "correlation_threshold": 0.01,
    }
    assert request.configuration["priors"] == {
        "lambda_prior_bounds": [-1.5, 1.5],
        "p_prior_bounds": [1.0, 1.0],
        "std_prior_factor": 3.0,
    }


@pytest.mark.parametrize(
    ("analysis_type", "configuration", "message"),
    [
        ("unknown", {}, "analysis_type"),
        (
            "event_count_donor_ignorant",
            {"unexpected": True},
            "unknown fields",
        ),
        (
            "event_count_donor_ignorant",
            {"models": ["heterogeneous_history_dependent"]},
            "invalid",
        ),
        (
            "trajectory_donor_ignorant",
            {"models": ["hetero3"]},
            "invalid",
        ),
        (
            "event_count_donor_ignorant",
            {"sampler": {"particles": 64, "chains": 1, "cores": 2}},
            "cannot exceed",
        ),
        (
            "event_count_donor_ignorant",
            {"priors": {"donor_deviation_prior": [0.3, 0.3, 1.0]}},
            "only valid",
        ),
        (
            "trajectory_donor_ignorant",
            {"priors": {"n_quad": 81}},
            "between 5 and 80",
        ),
    ],
)
def test_invalid_tagged_configurations_fail_closed(
    analysis_type: str,
    configuration: dict,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        validate_configuration(analysis_type, configuration)


def test_runner_normalizes_server_side_and_verifies_artifacts(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    runtime = FakeRuntime(adapter)
    progress: list[dict] = []
    payload = b"cell_id,condition,count\ncell_1,Control,2\n"

    result = run_request(
        analysis_type="event_count_donor_ignorant",
        dataset_file=BytesIO(payload),
        configuration={"models": ["homo"]},
        progress=progress.append,
        output_dir=tmp_path,
        runtime=runtime,
    )

    assert runtime.requested == ["event_count_donor_ignorant"]
    assert adapter.payload == payload
    assert adapter.configuration == result["configuration"]
    assert result["normalization"] == {"conditions": 1, "rows": 1}
    assert result["summary"] == {"best_model": "homo", "canonical_rows": 1}
    assert result["artifacts"] == [
        {
            "role": "model-evidence",
            "path": "results/evidence.csv",
            "media_type": "text/csv",
                "bytes": 30,
                "sha256": "af66656cff53ed5c0ba8b00f1192c1da46d9d61e2baf1280f1c34cdc732fcb06",
        }
    ]
    assert progress == [
        {
            "schema_version": 1,
            "sequence": 1,
            "phase": "sampling",
            "fraction": 0.25,
        },
        {
            "schema_version": 1,
            "sequence": 2,
            "phase": "sampling",
            "fraction": 0.75,
            "beta": 0.5,
        },
    ]
    assert result["_artifacts"] == [
        {
            "role": "model-evidence",
            "filename": "evidence.csv",
            "content_type": "text/csv",
            "payload": b"model,log_evidence\nhomo,-10.2\n",
            "shareable": True,
        }
    ]


def test_flat_api_configuration_is_canonicalized() -> None:
    request = validate_configuration(
        "event_count_donor_ignorant",
        {
            "models": ["homo"],
            "particles": 64,
            "chains": 1,
            "cores": 1,
            "observation_time": 1,
            "hdi_probability": 0.9,
        },
    )

    assert request.configuration["models"] == ["homo"]
    assert request.configuration["sampler"]["particles"] == 64
    assert request.configuration["hdi_probability"] == 0.9


def test_service_runner_preserves_native_progress_and_artifact_payloads(monkeypatch) -> None:
    def fake_request(**kwargs):
        kwargs["progress"](
            {
                "schema_version": 1,
                "sequence": 1,
                "phase": "sampling",
                "fraction": 0.7,
                "chain": {"index": 0, "total": 1, "stage": 3, "beta": 0.5},
            }
        )
        return {
            "schema_version": 1,
            "analysis_type": "event_count_donor_ignorant",
            "input_sha256": "a" * 64,
            "configuration": {"models": ["homo"]},
            "normalization": {"rows": 5},
            "summary": {"evidence": []},
            "artifacts": [{"role": "model-evidence"}],
            "_artifacts": [{"role": "model-evidence", "payload": b"csv"}],
        }

    monkeypatch.setattr(barracuda_runner, "run_request", fake_request)
    updates: list[dict] = []
    output = run_analysis(
        analysis_type="event_count_donor_ignorant",
        dataset_file=BytesIO(b"ignored"),
        configuration={},
        progress=updates.append,
    )

    assert updates[0]["chain"]["beta"] == 0.5
    assert updates[0]["fraction"] == 0.7
    assert output["_artifacts"][0]["payload"] == b"csv"
    assert "artifacts_persisted" not in output


def test_artifact_path_traversal_and_symlinks_are_rejected(tmp_path: Path) -> None:
    traversal = FakeAdapter(artifact_path="../outside.csv")
    with pytest.raises(ContractError, match="stay below"):
        run_request(
            analysis_type="event_count_donor_ignorant",
            dataset_file=BytesIO(b"cell_id,count\na,1\n"),
            configuration={},
            progress=lambda _event: None,
            output_dir=tmp_path / "traversal",
            runtime=FakeRuntime(traversal),
        )

    class SymlinkAdapter(FakeAdapter):
        def run(self, data, *, configuration, progress, output_dir):
            target = output_dir / "target.csv"
            target.write_text("safe")
            (output_dir / "linked.csv").symlink_to(target)
            return {
                "summary": {},
                "artifacts": [
                    {
                        "role": "model-evidence",
                        "path": "linked.csv",
                        "media_type": "text/csv",
                    }
                ],
            }

    with pytest.raises(ContractError, match="regular file"):
        run_request(
            analysis_type="event_count_donor_ignorant",
            dataset_file=BytesIO(b"cell_id,count\na,1\n"),
            configuration={},
            progress=lambda _event: None,
            output_dir=tmp_path / "symlink",
            runtime=FakeRuntime(SymlinkAdapter()),
        )


def test_non_json_summary_and_oversized_input_are_rejected(tmp_path: Path) -> None:
    class InvalidSummaryAdapter(FakeAdapter):
        def run(self, data, *, configuration, progress, output_dir):
            return {"summary": {"bad": float("nan")}, "artifacts": []}

    with pytest.raises(ContractError, match="finite JSON"):
        run_request(
            analysis_type="event_count_donor_ignorant",
            dataset_file=BytesIO(b"cell_id,count\na,1\n"),
            configuration={},
            progress=lambda _event: None,
            output_dir=tmp_path / "invalid-json",
            runtime=FakeRuntime(InvalidSummaryAdapter()),
        )

    class OversizedStream:
        def seek(self, _offset):
            return None

        def read(self, size):
            return b"x" * size

    with pytest.raises(ContractError, match="exceeds"):
        run_request(
            analysis_type="event_count_donor_ignorant",
            dataset_file=OversizedStream(),
            configuration={},
            progress=lambda _event: None,
            output_dir=tmp_path / "oversized",
            runtime=FakeRuntime(FakeAdapter()),
        )
