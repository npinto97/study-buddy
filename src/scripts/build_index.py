from app.indexer import index_data, save_index

if __name__ == "__main__":
    data_dir = "data/"
    print("Starting indexing...")
    index = index_data(data_dir)
    save_index(index)
    print(f"Indexing complete. Indexed {len(index)} files.")
