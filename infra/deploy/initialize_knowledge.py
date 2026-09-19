"""Initialize this fresh installation; never reset an already populated index."""
import json


def main():
    from back.infrastructure.search.opensearch import ensure_search_backend, get_vector_store, rebuild_tenant_vector_store
    from back.tenant.service import init_tenant_system, get_all_knowledge_sources, update_knowledge_source_status
    init_tenant_system()
    ensure_search_backend()
    store = get_vector_store("default")
    if store.count() == 0:
        store = rebuild_tenant_vector_store("default")
    else:
        print("Existing non-empty index retained; validating only", flush=True)
    images = store.get(where={"content_type": {"$eq": "image_semantic"}})
    orders = sorted({int(metadata["image_order"]) for metadata in images["metadatas"]})
    assert orders == list(range(1, 12)), f"Missing reference image semantics: {orders}"
    sources = []
    for source in get_all_knowledge_sources("default"):
        count = len(store.get_source_ids(source.source_id))
        assert count > 0, f"Empty indexed source: {source.source_id}"
        update_knowledge_source_status(source.source_id, "ready", chunk_count=count)
        sources.append({"source_id": source.source_id, "chunks": count})
    assert len(sources) == 2
    print(json.dumps({"status": "PASS", "total_chunks": store.count(), "image_orders": orders, "sources": sources}), flush=True)


if __name__ == "__main__":
    main()
