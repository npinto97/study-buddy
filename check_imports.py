import sys
import os

# Add the current directory to sys.path
sys.path.append(os.getcwd())

print("Checking all imports...")

modules_to_check = [
    "pydub",
    "streamlit",
    "PIL",
    "streamlit_float",
    "streamlit_extras.bottom_container",
    "study_buddy.agent",
    "study_buddy.utils.tools",
    "yaml",
    "study_buddy.vectorstore_pipeline.external_resources_handler"
]

for module in modules_to_check:
    try:
        print(f"Importing {module}...")
        if module == "study_buddy.agent":
            from study_buddy.agent import compiled_graph
        elif module == "study_buddy.utils.tools":
            from study_buddy.utils.tools import AudioProcessor
        elif module == "study_buddy.vectorstore_pipeline.external_resources_handler":
            from study_buddy.vectorstore_pipeline.external_resources_handler import extract_text_from_url
        else:
            __import__(module)
        print(f"Success: {module}")
    except Exception as e:
        print(f"Error importing {module}: {e}")

print("Import check complete.")
