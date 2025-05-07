from study_buddy.utils.tools import summarize_tool
import os


file_path = "C:/Users/Ningo/Desktop/basic-text.pdf"
print(f"Verifica percorso del file: {os.path.exists(file_path)}")

result = summarize_tool.invoke(file_path)
print(result)
