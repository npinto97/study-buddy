print("Starting imports...")
try:
    print("Importing config...")
    from study_buddy import config
    print("Config imported.")

    print("Importing embeddings...")
    from study_buddy.utils import embeddings
    print("Embeddings imported.")

    print("Importing vector_store_builder...")
    from study_buddy.vectorstore_pipeline import vector_store_builder
    print("Vector store builder imported.")

    print("Importing agent...")
    from study_buddy import agent
    print("Agent imported.")

except Exception as e:
    print(f"Caught exception: {e}")
    import traceback
    traceback.print_exc()

print("Imports finished.")
