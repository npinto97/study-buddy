import os
from pathlib import Path

RAW_DIR = Path("data/raw")

def find_siia_syllabus():
    found = []
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if "SIIA_syllabus" in file:
                found.append(os.path.join(root, file))
    
    if found:
        print("Found SIIA syllabus files:")
        for f in found:
            print(f)
    else:
        print("SIIA syllabus NOT found.")

if __name__ == "__main__":
    find_siia_syllabus()
