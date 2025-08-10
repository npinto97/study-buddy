from study_buddy.utils.tools import retrieve_tool

def main():
    query = "Professore Semeraro giovanni"
    result, docs = retrieve_tool(query)

    print("=== Documenti recuperati ===")
    for i, doc in enumerate(docs, start=1):
        print(f"\nDocumento {i}:")
        print(f"Metadata: {doc.metadata}")
        print(f"Contenuto: {doc.page_content}")

if __name__ == "__main__":
    main()
