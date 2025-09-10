from study_buddy.utils.tools import retrieve_knowledge

def main():
    query = "Professore Semeraro giovanni"
    result, docs = retrieve_knowledge(query)

    print("=== Documenti recuperati ===")
    for i, doc in enumerate(docs, start=1):
        print(f"\nDocumento {i}:")
        print(f"Metadata: {doc.metadata}")
        print(f"Contenuto: {doc.page_content}")

if __name__ == "__main__":
    main()
