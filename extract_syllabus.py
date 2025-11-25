import sys
import os
sys.path.append(os.getcwd())

from study_buddy.utils.tools import DocumentProcessor

# Extract text from the syllabus
processor = DocumentProcessor()
syllabus_path = r"C:\Users\Utente\OneDrive - Università degli Studi di Bari\Universita\Magistrale\II Anno\I Semestre\Semantics\Project\univox\data\raw\syllabuses\SIIA_syllabus.pdf"

print(f"Extracting text from: {syllabus_path}")
text = processor.extract_text(syllabus_path)
print("\n" + "="*80)
print("SYLLABUS CONTENT:")
print("="*80)
print(text)
print("\n" + "="*80)

# Search for email patterns
import re
emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
print(f"\nFound {len(emails)} email(s):")
for email in emails:
    print(f"  - {email}")
