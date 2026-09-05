import json
from pathlib import Path

from doc_intel.dataset.generate import Manifest, generate_dataset, load_truth


def test_generate_writes_documents_truth_and_manifest(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path, n=9, seed=1)
    assert len(manifest.documents) == 9
    for doc in manifest.documents:
        folder = tmp_path / doc.id
        assert (folder / "page.pdf").exists()
        assert (folder / "photo.jpg").stat().st_size > 10_000
        truth = load_truth(folder)
        assert truth.language.value == doc.language
        meta = json.loads((folder / "meta.json").read_text())
        assert meta["degradation"]["level"] == doc.level
    reloaded = Manifest.model_validate_json((tmp_path / "manifest.json").read_text())
    assert reloaded == manifest


def test_planted_issues_are_recorded(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path, n=10, seed=2)
    by_id = {d.id: d for d in manifest.documents}
    mismatch = [d for d in manifest.documents if "totals_mismatch" in d.expected_issues]
    assert len(mismatch) == 1  # index 3; the next would be 10
    truth = load_truth(tmp_path / mismatch[0].id)
    line_sum = sum((item.total.value or 0) for item in truth.line_items)
    assert truth.totals.grand_total.value != line_sum + (truth.totals.tax_total.value or 0)

    duplicates = [d for d in manifest.documents if "duplicate_number" in d.expected_issues]
    assert len(duplicates) == 2
    numbers = {load_truth(tmp_path / d.id).number.value for d in duplicates}
    assert len(numbers) == 1
    assert by_id[duplicates[0].id].id < by_id[duplicates[1].id].id
