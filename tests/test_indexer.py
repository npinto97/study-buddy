from app.indexer import index_data

def test_indexing():
    index = index_data("data/")
    assert len(index) > 0  # Should find files
    for entry in index:
        assert "file_type" in entry
        assert "title" in entry
        assert "path" in entry
