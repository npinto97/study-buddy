try:
    import unstructured
    print("Successfully imported unstructured")
except ImportError as e:
    print(f"Failed to import unstructured: {e}")

try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
    print("Successfully imported UnstructuredPDFLoader")
except ImportError as e:
    print(f"Failed to import UnstructuredPDFLoader: {e}")

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    print("Successfully imported UnstructuredWordDocumentLoader")
except ImportError as e:
    print(f"Failed to import UnstructuredWordDocumentLoader: {e}")

try:
    from langchain_community.document_loaders import NotebookLoader
    print("Successfully imported NotebookLoader")
except ImportError as e:
    print(f"Failed to import NotebookLoader: {e}")

try:
    from langchain_community.document_loaders import UnstructuredFileLoader
    print("Successfully imported UnstructuredFileLoader")
except ImportError as e:
    print(f"Failed to import UnstructuredFileLoader: {e}")
