# Booksmart :books:

Booksmart is a literary analysis tool that uses Natural Language Processing (NLP) to automatically extract and analyze key elements from EPUB books, including characters, themes, motifs, plotlines, and character relationships.

## Features

- **EPUB Processing**: Extract and analyze text from EPUB format books
- **Character Detection**: Identify main characters and their frequency in the text
- **Theme Analysis**: Discover potential themes through noun and concept analysis
- **Motif Detection**: Find recurring elements throughout the book
- **Plotline Extraction**: Generate plotline elements using subject-verb-object patterns
- **Character Relationship Mapping**: Analyze how characters relate to each other
- **GPU Acceleration**: Optional GPU support via PyTorch and spaCy for faster processing
- **Chapter-by-Chapter Analysis**: Option to analyze books by chapter

## Requirements

- Python 3.6+
- PyTorch (for GPU acceleration)
- spaCy
- spaCy language models (en_core_web_sm, en_core_web_md, en_core_web_lg, or en_core_web_trf)

## Installation

```bash
# Install required packages
pip install ebooklib beautifulsoup4 spacy spacy[cuda12x] networkx scikit-learn

# Install spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf  # For transformer-based model (optional but more accurate)
```

## Usage
### Basic Usage
```bash
python epub_analyzer.py path/to/your/book.epub
```

### Advanced Options
```bash
python epub_analyzer.py [books.epub ...] --model en_core_web_trf --output-dir analysis_results --by-chapter
```

### Command Line Arguments
- `files`: One or more EPUB files to analyze
- `--output-dir`: Directory for output files (default: 'analysis_output')
- `by-chapter`: Analyze book chapter by chapter
- `--model`:  SpaCy model to use (choices: 'en_core_web_sm', 'en_core_web_md', 'en_core_web_lg', 'en_core_web_trf')
- `--disable-gpu`: Disable GPU acceleration even if available
- `--batch-size`: Batch size for transformer processing (reduce if running out of GPU memory)
- `--debug`: Enable debug output

## GPU Support
The tool will automatically detect and use GPU acceleration if available. Right now, it natively supports CUDA 12.