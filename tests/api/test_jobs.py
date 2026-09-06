from doc_intel.api.jobs import InMemoryJobStore, JobStatus


async def test_create_then_get() -> None:
    store = InMemoryJobStore()
    job = await store.create(filename="inv.jpg", mime="image/jpeg", size_bytes=1234)
    assert job.status is JobStatus.QUEUED
    assert await store.get(job.id) == job
    assert await store.get("missing") is None


async def test_list_is_oldest_first() -> None:
    store = InMemoryJobStore()
    first = await store.create("a.pdf", "application/pdf", 1)
    second = await store.create("b.pdf", "application/pdf", 2)
    assert [job.id for job in await store.list()] == [first.id, second.id]


async def test_set_status_replaces_the_job() -> None:
    store = InMemoryJobStore()
    job = await store.create("a.pdf", "application/pdf", 1)
    failed = await store.set_status(job.id, JobStatus.FAILED, error="boom")
    assert failed.status is JobStatus.FAILED
    assert failed.error == "boom"
    assert await store.get(job.id) == failed
