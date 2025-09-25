# `data/`

This directory contains the educational content metadata and data files used by the Study Buddy system for semantic search and information retrieval.

## Directory Structure

```
data/
├── README.md                   # This file
├── metadata/                   # Course metadata (JSON files)
│   ├── SIIA/                   # Course: Semantic Information and Intelligent Applications
│   │   ├── course_metadata.json
│   │   ├── lesson_01.json
│   │   ├── lesson_02.json
│   │   └── ...
│   ├── MRI/                    # Course: Methods for Information Retrieval
│   │   ├── course_metadata.json
│   │   ├── lesson_01.json
│   │   ├── lesson_12.json
│   │   └── ...
│   └── [other_courses]/
└── raw_files/                  # Actual content files (PDFs, videos, etc.)
    ├── slides/
    ├── books/
    ├── exercises/
    ├── references/
    └── ...
```

## Metadata Structure

Each course follows a **two-level hierarchy**:

1. **Course Level** (`course_metadata.json`)
   - Course name and description
   - Syllabus
   - Textbooks and bibliography
   - General course notes
   - List of lessons with references to their metadata

2. **Lesson Level** (`lesson_XX.json`)
   - Lesson title and keywords
   - Presentation slides
   - Academic references
   - Supplementary materials
   - Exercises and exam papers
   - Multimedia content (videos, images, audio)
   - External resources and web links

### Content Types Supported

The metadata system supports **12 different content types**:

| Type | Description | Example Files |
|------|-------------|---------------|
| `syllabus` | Course program and overview | `course_syllabus.pdf` |
| `book` | Textbooks and reference books | `textbook_chapter1.pdf` |
| `note` | General course notes | `course_notes.pdf` |
| `slide` | Presentation slides | `lesson01_slides.pdf` |
| `lesson_note` | Lesson-specific notes | `lesson01_notes.pdf` |
| `reference` | Academic papers and references | `research_paper.pdf` |
| `supplementary_material` | Additional learning materials | `extra_reading.pdf` |
| `exercise` | Practice exercises and exams | `midterm_exam.pdf` |
| `video` | Video lectures and tutorials | `lecture_recording.mp4` |
| `image` | Diagrams and illustrations | `algorithm_diagram.jpg` |
| `audio` | Audio recordings | `interview.mp3` |
| `external_resource` | Web links and online content | YouTube videos, articles |


### Understanding the Metadata

Here's what a typical course metadata looks like:

```json
{
  "course_name": "Name of the course",
  "description": "Advanced course on ...",
  "syllabus": "syllabus/course_syllabus.pdf",
  "books": [
    {
      "filename": "books/course_textbook.pdf",
      "title": "Course book title",
      "author": " Name Surname",
      "year": 2011,
      "isbn": "***-*****"
    }
  ],
  "lessons": [
    {
      "lesson_number": 1,
      "metadata": "lesson_01.json"
    }
  ]
}
```

And a lesson metadata example:

```json
{
  "lesson_number": 12,
  "title": "Practice Session",
  "keywords": ["Exercises", "Exam Papers", "Practice"],
  "slides": [],
  "exercises": [
    {
      "title": "Course_name Exam Papers Collection",
      "filename": [
        "exercises/exam_papers/Course_name_Exam_Paper_1.docx",
        "exercises/exam_papers/Course_name_Exam_Paper_2.doc"
      ],
      "description": "Collection of past exam papers and practice tests"
    }
  ],
  "external_resources": [
    {
      "type": "video",
      "title": "The Zipf Mystery",
      "url": "https://www.youtube.com/watch?v=fCn8zs912OE",
      "description": "Power law and Pareto Principle explanation"
    }
  ]
}
```

## How It Works

1. **Metadata Parsing**: The system scans all course directories and processes the JSON metadata files
2. **Content Extraction**: Each file reference is resolved to actual content in the `raw_files/` directory
3. **Normalization**: All content is transformed into a flat structure with full context information
4. **Indexing**: The processed metadata feeds into the semantic search system


### Adding New Content

1. **Create course directory**: `mkdir metadata/YOUR_COURSE`
2. **Add course metadata**: Create `course_metadata.json` following the schema
3. **Add lesson metadata**: Create `lesson_XX.json` files for each lesson
4. **Place actual files**: Put content files in appropriate `raw_files/` subdirectories
5. **Run parser**: Execute the metadata parser to update the system
