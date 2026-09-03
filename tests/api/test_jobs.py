from doc_intel.api.jobs import InMemoryJobStore, JobStatus


def test_create_then_get() -> None:
    store = InMemoryJobStore()
    job = store.create(filename="inv.jpg", mime="image/jpeg", size_bytes=1234)
    assert job.status is JobStatus.QUEUED
    assert store.get(job.id) == job
    assert store.get("missing") is None


def test_list_is_oldest_first() -> None:
    store = InMemoryJobStore()
    first = store.create("a.pdf", "application/pdf", 1)
    second = store.create("b.pdf", "application/pdf", 2)
    assert [job.id for job in store.list()] == [first.id, second.id]


def test_set_status_replaces_the_job() -> None:
    store = InMemoryJobStore()
    job = store.create("a.pdf", "application/pdf", 1)
    failed = store.set_status(job.id, JobStatus.FAILED, error="boom")
    assert failed.status is JobStatus.FAILED
    assert failed.error == "boom"
    assert store.get(job.id) == failed
