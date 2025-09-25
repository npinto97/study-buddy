import json
from study_buddy.config import PARSED_COURSES_DATA_FILE
from study_buddy.vectorstore_pipeline.document_loader import scan_directory_for_new_documents


def test_document_loader():
    """Testa il document loader leggendo i file da parsed_course_data.json"""

    # Verifica che il file esista
    if not PARSED_COURSES_DATA_FILE.exists():
        print(f"Errore: {PARSED_COURSES_DATA_FILE} non trovato!")
        return

    # Carica il JSON
    with open(PARSED_COURSES_DATA_FILE, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

    print(f"Trovati {len(parsed_data)} file da processare...\n")

    # Simuliamo un set di hash di file già processati (vuoto per il test)
    processed_hashes = set()

    # Eseguiamo il loader
    new_docs, _ = scan_directory_for_new_documents(processed_hashes, PARSED_COURSES_DATA_FILE)

    # Output dei risultati
    print("\n📄 Documenti estratti:")
    for doc in new_docs:
        # print(f"- Path: {doc.metadata.get('path', 'N/A')}")
        # print(f"  Corso: {doc.metadata.get('course_name', 'N/A')}")
        # print(f"  Tipo: {doc.metadata.get('type', 'N/A')}")
        print(f"  METADATA: {doc.metadata}")
        print(f"  CONTENUTO: {doc.page_content[:200]}...\n")  # Mostra solo un estratto

    print(f"✅ Test completato! {len(new_docs)} documenti caricati.")


# Avvia il test
if __name__ == "__main__":
    test_document_loader()
