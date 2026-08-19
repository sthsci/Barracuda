from __future__ import annotations

import json
import re
from pathlib import Path

from webapp.pages.notebooks import COLAB_ROOT, NOTEBOOKS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
PRIVATE_NAMES = (
    "NT-No Treatment.csv",
    "RTX-Rituximab.csv",
    "Bispecific-Bispecific.csv",
    "Donor-info_NT.csv",
    "Donor-info_RTX.csv",
    "Donor-info_Bispecific.csv",
    ".numbers",
    ".prism",
)
LOCAL_PATH = re.compile(
    r"/Users/|/home/|/private/|/tmp/|/var/folders/|/Volumes/|(?<![A-Za-z])[A-Za-z]:[\\/]"
)


def test_published_colab_notebooks_are_clean_and_portable() -> None:
    registered = [item["path"] for item in NOTEBOOKS]
    on_disk = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in NOTEBOOK_ROOT.glob("*.ipynb")
    }

    assert len(registered) == len(set(registered)) == 7
    assert set(registered) == on_disk

    for item in NOTEBOOKS:
        relative_path = item["path"]
        path = PROJECT_ROOT / relative_path
        notebook = json.loads(path.read_text(encoding="utf-8"))

        assert notebook["nbformat"] == 4, relative_path
        assert notebook["nbformat_minor"] >= 5, relative_path
        assert isinstance(notebook["cells"], list) and notebook["cells"], relative_path

        metadata = notebook["metadata"]
        assert metadata["kernelspec"]["name"] == "python3", relative_path
        assert metadata["kernelspec"]["display_name"] == "Python 3", relative_path
        assert metadata["language_info"]["name"] == "python", relative_path
        assert metadata["language_info"]["version"] == "3.12", relative_path

        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(isinstance(cell_id, str) and cell_id.strip() for cell_id in cell_ids), relative_path
        assert len(cell_ids) == len(set(cell_ids)), relative_path

        first_markdown = next(
            cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        expected_colab_url = f"{COLAB_ROOT}/{relative_path}"
        assert f"]({expected_colab_url})" in "".join(first_markdown["source"]), relative_path

        sources = []
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            sources.append(source)
            if cell["cell_type"] != "code":
                continue

            assert cell["execution_count"] is None, relative_path
            assert cell["outputs"] == [], relative_path
            assert not any(
                line.lstrip().startswith(("%", "!")) for line in source.splitlines()
            ), relative_path
            compile(source, f"{relative_path}:{cell['id']}", "exec")

        full_source = "\n".join(sources)
        assert not LOCAL_PATH.search(full_source), relative_path
        assert not any(name in full_source for name in PRIVATE_NAMES), relative_path

        needs_inference_guard = (
            item["category"] == "Analysis" or relative_path.endswith("_tutorial.ipynb")
        )
        if needs_inference_guard:
            assert re.search(r"(?m)^RUN_INFERENCE\s*=\s*False\b", full_source), relative_path

        if relative_path == "notebooks/00_run_the_orca_web_app.ipynb":
            assert re.search(r"(?m)^RUN_WEB_APP\s*=\s*False\b", full_source)
            assert 'jupyter_mode="inline"' in full_source
