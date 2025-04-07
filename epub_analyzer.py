#!/usr/bin/env python3
"""
EPUB Literary Analysis Tool
Analyzes EPUB files to extract themes, characters, motifs and plotlines.
"""

import argparse
import os
import json
import re
from collections import Counter, defaultdict
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import spacy
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze EPUB files for literary elements')
    parser.add_argument('files', nargs='+', help='One or more EPUB files to analyze')
    parser.add_argument('--output-dir', default='analysis_output', help='Directory for output files')
    parser.add_argument('--by-chapter', action='store_true', help='Analyze book chapter by chapter')
    parser.add_argument('--model', default='en_core_web_sm', 
                        choices=['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg', 'en_core_web_trf'],
                        help='SpaCy model to use (larger models are more accurate but slower, transformer model is most accurate)')
    parser.add_argument('--disable-gpu', action='store_true', 
                        help='Disable GPU acceleration even if available')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for transformer processing (reduce if running out of GPU memory)')
    return parser.parse_args()

def extract_text_from_epub(epub_path, by_chapter=False):
    """Extract and organize text content from an EPUB file."""
    print(f"Extracting text from {epub_path}...")
    book = epub.read_epub(epub_path)
    
    # Get book metadata
    title = "Unknown"
    author = "Unknown"
    
    for item in book.get_metadata('DC', 'title'):
        title = item[0]
        break
        
    for item in book.get_metadata('DC', 'creator'):
        author = item[0]
        break
    
    if by_chapter:
        chapters = []
        chapter_titles = []
        
        # Extract chapters from spine
        spine_items = []
        for itemref in book.spine:
            item_id = itemref[0]
            for item in book.get_items():
                if item.id == item_id:
                    spine_items.append(item)
        
        for i, item in enumerate(spine_items):
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8', errors='replace')
                soup = BeautifulSoup(content, 'html.parser')
                
                # Try to find chapter title
                chapter_title = f"Chapter {i+1}"
                heading = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if heading:
                    chapter_title = heading.get_text().strip()
                
                # Extract text
                text = soup.get_text()
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text) > 200:  # Only include substantial chapters
                    chapters.append(text)
                    chapter_titles.append(chapter_title)
        
        # If we couldn't extract proper chapters, just return the whole text
        if len(chapters) <= 1:
            print("Couldn't identify distinct chapters, processing as a single text.")
            return extract_text_from_epub(epub_path, by_chapter=False)
            
        return {
            "title": title,
            "author": author,
            "chapters": chapters,
            "chapter_titles": chapter_titles,
            "full_text": "\n\n".join(chapters)
        }
    else:
        text_content = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8', errors='replace')
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text()
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                text_content.append(text)
        
        full_text = "\n".join(text_content)
        return {
            "title": title,
            "author": author,
            "full_text": full_text,
            "chapters": [full_text],
            "chapter_titles": ["Complete Text"]
        }

def check_gpu_availability():
    """Check if GPU is available for spaCy's transformers."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            return True, device_count, device_names
        else:
            return False, 0, []
    except ImportError:
        return False, 0, []

def detect_characters(text, nlp):
    """Detect and analyze characters in the text."""
    doc = nlp(text[:500000])  # Limit for memory usage
    
    # Extract named entities that are people
    characters = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    # Count frequency
    character_counts = Counter(characters)
    
    # Return characters mentioned multiple times
    return [{"name": name, "mentions": count} 
            for name, count in character_counts.most_common(50) 
            if count > 2 and len(name) > 1]

def detect_themes(text, nlp):
    """Detect potential themes in the text."""
    # Process a reasonable chunk of text
    doc = nlp(text[:200000])
    
    # Extract noun chunks and important words
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks 
                  if len(chunk.text) > 3 and not all(token.is_stop for token in chunk)]
    
    # Count frequency
    theme_counts = Counter(noun_chunks)
    
    # Get abstract nouns (potential themes)
    abstract_nouns = [token.lemma_.lower() for token in doc 
                     if token.pos_ == "NOUN" and not token.is_stop and len(token.text) > 3]
    
    # Combine with important adjectives
    important_adj = [token.lemma_.lower() for token in doc 
                     if token.pos_ == "ADJ" and not token.is_stop and len(token.text) > 3]
    
    # Combine concepts
    concepts = Counter(abstract_nouns + important_adj)
    
    # Return top concepts as potential themes
    return [{"theme": theme, "mentions": count} 
            for theme, count in concepts.most_common(30)]

def detect_motifs(chapters, nlp):
    """Detect recurring motifs across chapters."""
    if len(chapters) <= 1:
        # For single chapter, look for repeated phrases
        doc = nlp(chapters[0][:200000])
        phrases = []
        for sent in doc.sents:
            if len(sent.text) > 10 and len(sent.text) < 80:
                phrases.append(sent.text)
        
        # Count frequent phrases
        phrase_counter = Counter(phrases)
        return [{"motif": motif, "occurrences": count} 
                for motif, count in phrase_counter.most_common(20) if count > 1]
    
    # For multi-chapter books, use TF-IDF to find important phrases
    tfidf = TfidfVectorizer(ngram_range=(2, 4), max_features=100)
    
    # If chapters are very long, use only the first part of each
    processed_chapters = [chapter[:50000] for chapter in chapters]
    
    tfidf_matrix = tfidf.fit_transform(processed_chapters)
    feature_names = tfidf.get_feature_names_out()
    
    # Extract top phrases from each chapter
    chapter_phrases = []
    for i in range(len(processed_chapters)):
        feature_index = tfidf_matrix[i,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        top_phrases = sorted([(feature_names[i], s) for (i, s) in tfidf_scores], key=lambda x: x[1], reverse=True)
        chapter_phrases.append([phrase for phrase, score in top_phrases[:10]])
    
    # Find recurring phrases across chapters
    all_phrases = [phrase for phrases in chapter_phrases for phrase in phrases]
    phrase_counts = Counter(all_phrases)
    
    return [{"motif": motif, "occurrences": count} 
            for motif, count in phrase_counts.most_common(20) if count > 1]

def detect_plotlines(text, nlp):
    """Detect potential plotlines using verb phrases and events."""
    # Process a section of the text
    doc = nlp(text[:200000])
    
    # Extract subject-verb-object patterns
    plot_events = []
    for sent in doc.sents:
        subjects = []
        verbs = []
        objects = []
        
        for token in sent:
            # Find subjects
            if token.dep_ in ["nsubj", "nsubjpass"] and not token.is_stop:
                subjects.append(token.text)
            
            # Find main verbs
            if token.pos_ == "VERB" and token.dep_ not in ["aux", "auxpass"]:
                verbs.append(token.lemma_)
            
            # Find objects
            if token.dep_ in ["dobj", "pobj"] and not token.is_stop:
                objects.append(token.text)
        
        # Construct basic events
        if subjects and verbs:
            for subj in subjects:
                for verb in verbs:
                    if objects:
                        for obj in objects:
                            plot_events.append(f"{subj} {verb} {obj}")
                    else:
                        plot_events.append(f"{subj} {verb}")
    
    # Count frequency
    event_counts = Counter(plot_events)
    
    # Return common events as plotlines
    return [{"event": event, "mentions": count} 
            for event, count in event_counts.most_common(30) if count > 1]

def analyze_character_relationships(text, characters, nlp):
    """Analyze relationships between characters."""
    if not characters:
        return []
        
    # Create a graph of character co-occurrences
    graph = nx.Graph()
    
    # Add characters as nodes
    character_names = [c["name"] for c in characters]
    for char in character_names:
        graph.add_node(char)
    
    # Process text in chunks
    max_length = 100000
    text_chunks = [text[i:i+max_length] for i in range(0, min(len(text), 300000), max_length)]
    
    for chunk in text_chunks:
        doc = nlp(chunk)
        
        # Analyze each sentence for character co-occurrence
        for sent in doc.sents:
            sent_text = sent.text
            
            # Check which characters appear in this sentence
            chars_in_sent = []
            for char in character_names:
                if char in sent_text:
                    chars_in_sent.append(char)
            
            # Add edges between characters that co-occur
            for i, char1 in enumerate(chars_in_sent):
                for char2 in chars_in_sent[i+1:]:
                    if graph.has_edge(char1, char2):
                        graph[char1][char2]['weight'] += 1
                    else:
                        graph.add_edge(char1, char2, weight=1)
    
    # Extract relationships
    relationships = []
    for char1, char2, data in graph.edges(data=True):
        if data['weight'] > 1:  # Only include relationships mentioned multiple times
            relationships.append({
                "character1": char1,
                "character2": char2,
                "interactions": data['weight']
            })
    
    # Sort by interaction count
    relationships = sorted(relationships, key=lambda x: x['interactions'], reverse=True)
    
    return relationships[:30]  # Return top 30 relationships

def analyze_text(book_data, model_name, disable_gpu=False, batch_size=16):
    """Analyze text to identify literary elements."""
    print(f"Loading spaCy model '{model_name}'...")
    
    # Check for GPU if using transformer model
    if model_name == 'en_core_web_trf' and not disable_gpu:
        gpu_available, gpu_count, gpu_names = check_gpu_availability()
        if gpu_available:
            print(f"GPU acceleration enabled! Found {gpu_count} device(s):")
            for i, name in enumerate(gpu_names):
                print(f"  - GPU {i}: {name}")
            
            # Set PyTorch environment variables for better performance
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            
            # Load model with GPU support
            spacy.prefer_gpu()
            nlp = spacy.load(model_name)
            
            # Configure batch size for transformer
            if hasattr(nlp, 'transformer'):
                print(f"Setting transformer batch size to {batch_size}")
                nlp.transformer.batch_size = batch_size
        else:
            print("No GPU detected. Transformer model will run on CPU (much slower).")
            nlp = spacy.load(model_name)
    else:
        # For non-transformer models or when GPU is disabled
        nlp = spacy.load(model_name)
    
    # Configure spaCy for memory efficiency
    nlp.max_length = 1500000
    
    # Adjust processing for transformer model which handles text differently
    if model_name == 'en_core_web_trf':
        # For transformer models, we need to process text in smaller chunks
        nlp.max_length = 500000
    
    full_text = book_data["full_text"]
    chapters = book_data["chapters"]
    chapter_titles = book_data["chapter_titles"]
    
    print("Detecting characters...")
    characters = detect_characters(full_text, nlp)
    
    print("Analyzing themes...")
    themes = detect_themes(full_text, nlp)
    
    print("Identifying motifs...")
    motifs = detect_motifs(chapters, nlp)
    
    print("Extracting plotlines...")
    plotlines = detect_plotlines(full_text, nlp)
    
    print("Mapping character relationships...")
    relationships = analyze_character_relationships(full_text, characters, nlp)
    
    # Analyze each chapter
    chapter_analysis = []
    if len(chapters) > 1:
        print(f"Analyzing {len(chapters)} individual chapters...")
        for i, (chapter, title) in enumerate(zip(chapters, chapter_titles)):
            print(f"  Processing chapter {i+1}: {title}")
            chapter_chars = detect_characters(chapter, nlp)
            chapter_themes = detect_themes(chapter, nlp)
            
            chapter_analysis.append({
                "title": title,
                "characters": chapter_chars[:15],
                "themes": chapter_themes[:10]
            })
    
    return {
        "title": book_data["title"],
        "author": book_data["author"],
        "characters": characters,
        "themes": themes,
        "motifs": motifs,
        "plotlines": plotlines,
        "relationships": relationships,
        "chapters": chapter_analysis,
        "statistics": {
            "total_words": len(full_text.split()),
            "total_characters": len(characters),
            "total_chapters": len(chapters)
        }
    }

def save_analysis(analysis, book_path, output_dir):
    """Save analysis to structured files for RAG."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(book_path))[0]
    
    # Create a clean filename
    base_name = re.sub(r'[^\w\-\.]', '_', base_name)
    
    print(f"Saving analysis files to {output_dir}/")
    
    # Save each component to a separate file for RAG
    with open(f"{output_dir}/{base_name}_characters.json", 'w') as f:
        json.dump({
            "book": analysis["title"],
            "author": analysis["author"],
            "characters": analysis["characters"]
        }, f, indent=2)
    
    with open(f"{output_dir}/{base_name}_themes.json", 'w') as f:
        json.dump({
            "book": analysis["title"],
            "author": analysis["author"],
            "themes": analysis["themes"]
        }, f, indent=2)
    
    with open(f"{output_dir}/{base_name}_motifs.json", 'w') as f:
        json.dump({
            "book": analysis["title"],
            "author": analysis["author"],
            "motifs": analysis["motifs"]
        }, f, indent=2)
    
    with open(f"{output_dir}/{base_name}_plotlines.json", 'w') as f:
        json.dump({
            "book": analysis["title"],
            "author": analysis["author"],
            "plotlines": analysis["plotlines"]
        }, f, indent=2)
    
    with open(f"{output_dir}/{base_name}_relationships.json", 'w') as f:
        json.dump({
            "book": analysis["title"],
            "author": analysis["author"],
            "relationships": analysis["relationships"]
        }, f, indent=2)
    
    if analysis["chapters"]:
        with open(f"{output_dir}/{base_name}_chapters.json", 'w') as f:
            json.dump({
                "book": analysis["title"],
                "author": analysis["author"],
                "chapters": analysis["chapters"]
            }, f, indent=2)
    
    # Save a combined analysis
    with open(f"{output_dir}/{base_name}_complete_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

def main():
    args = parse_arguments()
    
    for epub_file in args.files:
        try:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {epub_file}")
            print(f"{'='*60}")
            
            # Extract text content
            book_data = extract_text_from_epub(epub_file, by_chapter=args.by_chapter)
            
            print(f"Book: '{book_data['title']}' by {book_data['author']}")
            print(f"Text length: {len(book_data['full_text'])} characters")
            print(f"Chapters identified: {len(book_data['chapters'])}")
            
            # Analyze the text
            analysis = analyze_text(book_data, args.model, args.disable_gpu, args.batch_size)
            
            # Report findings
            print("\nANALYSIS SUMMARY:")
            print(f"- Found {len(analysis['characters'])} main characters")
            print(f"- Identified {len(analysis['themes'])} potential themes")
            print(f"- Detected {len(analysis['motifs'])} recurring motifs")
            print(f"- Extracted {len(analysis['plotlines'])} plotline elements")
            print(f"- Mapped {len(analysis['relationships'])} character relationships")
            
            # Save analysis
            save_analysis(analysis, epub_file, args.output_dir)
            print(f"\nAnalysis for '{analysis['title']}' completed successfully.")
            
        except Exception as e:
            print(f"\nERROR processing {epub_file}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()