#!/usr/bin/env python3
"""
EPUB Literary Analysis Tool - Enhanced for RAG Operations
Analyzes EPUB files to extract deep literary elements for conversational AI.
Optimized for NVIDIA RTX 3090 GPUs.
"""
import argparse
import os
import json
import re
import time
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import torch
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import spacy
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze EPUB files for literary elements optimized for RAG')
    parser.add_argument('files', nargs='+', help='One or more EPUB files to analyze')
    parser.add_argument('--output-dir', default='analysis_output', help='Directory for output files')
    parser.add_argument('--by-chapter', action='store_true', help='Analyze book chapter by chapter')
    parser.add_argument('--model', default='en_core_web_lg', 
                        choices=['en_core_web_md', 'en_core_web_lg', 'en_core_web_trf'],
                        help='SpaCy model to use')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                        help='Sentence transformer model for embeddings')
    parser.add_argument('--chunk-size', type=int, default=2000,
                        help='Character length for processing chunks')
    parser.add_argument('--batch-size', type=int, default=48,
                        help='Batch size for GPU processing')
    parser.add_argument('--max-text-length', type=int, default=1000000,
                        help='Maximum text length to analyze')
    parser.add_argument('--extract-quotes', action='store_true', default=True,
                        help='Extract and attribute character quotes')
    parser.add_argument('--detect-sentiment', action='store_true', default=True,
                        help='Analyze emotional tone and sentiment')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision (FP16) for transformer models')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def extract_text_from_epub(epub_path, by_chapter=False):
    """Extract and organize text content from an EPUB file with improved chapter detection."""
    print(f"Extracting text from {epub_path}...")
    book = epub.read_epub(epub_path)
    
    # Get book metadata with fallbacks
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
        
        # Extract items from spine in order
        spine_items = []
        for itemref in book.spine:
            item_id = itemref[0]
            for item in book.get_items():
                if item.id == item_id:
                    spine_items.append(item)
        
        # Better chapter detection logic
        for i, item in enumerate(spine_items):
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8', errors='replace')
                soup = BeautifulSoup(content, 'html.parser')
                
                # Try multiple approaches to find chapter title
                chapter_title = f"Chapter {i+1}"
                
                # First look for heading elements
                heading = soup.find(['h1', 'h2', 'h3', 'h4'])
                if heading and heading.get_text().strip():
                    chapter_title = heading.get_text().strip()
                # Try looking for div with chapter/title classes
                elif soup.find('div', class_=lambda c: c and ('chapter' in c.lower() or 'title' in c.lower())):
                    div = soup.find('div', class_=lambda c: c and ('chapter' in c.lower() or 'title' in c.lower()))
                    chapter_title = div.get_text().strip()
                # Try common chapter patterns
                else:
                    text = soup.get_text()
                    chapter_match = re.search(r'(?:chapter|part|book)\s+[IVXLCDM\d]+(?:[^\n.]{1,50})?', text, re.IGNORECASE)
                    if chapter_match:
                        chapter_title = chapter_match.group(0).strip()
                
                # Extract text
                text = soup.get_text()
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Filter out TOC, copyright pages, etc.
                if len(text) > 500:  # More substantial threshold for chapters
                    chapters.append(text)
                    chapter_titles.append(chapter_title)
        
        # If we couldn't extract proper chapters, try alternate detection methods
        if len(chapters) <= 1:
            print("Trying alternate chapter detection method...")
            # Get all HTML items
            doc_items = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
            
            # Sort by length (chapters are often similar in length)
            item_lengths = [(item, len(item.get_content())) for item in doc_items]
            avg_items = [item for item, length in item_lengths if length > 1000 and length < 100000]
            
            if len(avg_items) > 1:
                chapters = []
                chapter_titles = []
                for i, item in enumerate(avg_items):
                    content = item.get_content().decode('utf-8', errors='replace')
                    soup = BeautifulSoup(content, 'html.parser')
                    text = soup.get_text()
                    text = re.sub(r'\s+', ' ', text).strip()
                    chapters.append(text)
                    chapter_titles.append(f"Chapter {i+1}")
            else:
                print("Still couldn't identify chapters, processing as a single text.")
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

def optimize_gpu_settings():
    """Configure optimal settings for NVIDIA RTX 3090."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
        
        print(f"GPU: {device_name} with {mem_total:.1f}GB total memory")
        print(f"Currently reserved: {mem_reserved:.1f}GB")
        
        # RTX 3090 specific optimizations
        if "3090" in device_name:
            # Set optimal CUDA settings for 3090
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matrix multiplications
            torch.backends.cudnn.allow_tf32 = True
            
            # Set environment variables
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            
            # Calculate optimal batch size (can be adjusted)
            optimal_batch_size = min(int(mem_total * 1.0), 64)  # 1GB of VRAM per batch unit, capped at 64
            
            print(f"Optimized settings for RTX 3090 enabled")
            print(f"Recommended batch size: {optimal_batch_size}")
            return {
                "use_gpu": True,
                "optimal_batch_size": optimal_batch_size,
                "device": device,
                "memory": mem_total
            }
    
    return {"use_gpu": False}

def preprocess_text_for_rag(text, max_length=1000000):
    """Preprocess text for RAG by cleaning and truncating."""
    # Limit text length for processing
    if len(text) > max_length:
        text = text[:max_length]
    
    # Clean text for better processing
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
    text = re.sub(r'\(.*?\)', '', text)  # Remove parentheses
    text = re.sub(r'_{2,}', '', text)  # Remove underscores
    text = re.sub(r'\.{2,}', '...', text)  # Normalize ellipses
    
    return text

def load_nlp_models(model_name, embedding_model, fp16=True):
    """Load NLP models with optimizations for GPU."""
    # Load spaCy model
    print(f"Loading spaCy model '{model_name}'...")
    if torch.cuda.is_available():
        spacy.prefer_gpu()
    
    nlp = spacy.load(model_name)
    
    # Configure for memory efficiency
    if model_name == 'en_core_web_trf':
        nlp.max_length = 514 * 128  # Transformer context size * batch
    else:
        nlp.max_length = 1500000
    
    # Load sentence transformer model for embeddings
    print(f"Loading sentence transformer model '{embedding_model}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure FP16 for efficiency on GPU
    fp16_setting = fp16 and device == "cuda"
    
    # Load the model
    st_model = SentenceTransformer(embedding_model, device=device)
    if fp16_setting:
        st_model.half()  # Convert to FP16 for faster inference
    
    return nlp, st_model

def split_text_into_chunks(text, size=2000, overlap=200):
    """Split text into overlapping chunks for processing."""
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i + size]
        if len(chunk) > 100:  # Avoid tiny chunks
            chunks.append(chunk)
    return chunks

def extract_characters_advanced(text, nlp, st_model, batch_size=32):
    """
    Advanced character extraction with coreference and dialogue analysis.
    Uses both NER and custom patterns to identify characters.
    """
    print(f"Extracting characters with advanced techniques...")
    
    # Split into manageable chunks for GPU processing
    chunks = split_text_into_chunks(text, size=5000)
    print(f"Processing {len(chunks)} chunks for character detection...")
    
    # Process in batches using pipe for GPU efficiency
    all_entities = []
    
    # First pass: identify named entities
    for batch in tqdm([chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)], 
                      desc="Finding character entities"):
        docs = list(nlp.pipe(batch))
        
        for doc in docs:
            # Extract person entities
            entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
            all_entities.extend([(ent.text, doc.text[max(0, ent.start_char-100):min(len(doc.text), ent.end_char+100)]) 
                               for ent in entities])
            
            # Look for honorifics that might indicate characters
            honorific_pattern = r'(?:Mr|Mrs|Miss|Ms|Dr|Prof|Sir|Lady|Lord)\.\s+[A-Z][a-z]+'
            for match in re.finditer(honorific_pattern, doc.text):
                span = match.span()
                entity_text = match.group(0)
                context = doc.text[max(0, span[0]-100):min(len(doc.text), span[1]+100)]
                all_entities.append((entity_text, context))
    
    # Count and cluster similar character mentions
    character_counts = Counter([e[0] for e in all_entities])
    
    # Filter to significant mentions
    potential_characters = [char for char, count in character_counts.items() 
                           if count >= 2 and len(char) > 1]
    
    # Create embeddings for clustering
    if potential_characters:
        embeddings = st_model.encode(potential_characters, show_progress_bar=False)
        
        # Cluster similar character mentions
        clustering = DBSCAN(eps=0.25, min_samples=1).fit(embeddings)
        
        # Group by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            clusters[label].append(potential_characters[i])
        
        # Get primary name for each cluster (longest name with highest count)
        character_clusters = []
        for cluster_id, names in clusters.items():
            # Choose canonical name by scoring length and frequency
            name_scores = [(name, len(name) * 0.3 + character_counts[name] * 0.7) for name in names]
            canonical_name = max(name_scores, key=lambda x: x[1])[0]
            
            # Sum mentions of all variations
            total_mentions = sum(character_counts[name] for name in names)
            
            # Get contexts for this character
            contexts = [context for name, context in all_entities if name in names]
            sample_contexts = contexts[:5] if contexts else []
            
            character_clusters.append({
                "name": canonical_name,
                "variants": names,
                "mentions": total_mentions,
                "contexts": sample_contexts
            })
        
        # Sort by mention count
        character_clusters.sort(key=lambda x: x["mentions"], reverse=True)
        
        # Extract dialogue patterns for major characters
        extract_character_quotes(text, character_clusters[:10])
        
        return character_clusters[:30]  # Return top 30 characters
    
    return []

def extract_character_quotes(text, characters):
    """Extract representative quotes for major characters."""
    for character in characters:
        # Pattern to find dialogue attributed to character
        character_name = re.escape(character["name"])
        # Look for patterns like: "Quote," said Character or Character said, "Quote"
        patterns = [
            rf'["\'](.*?)["\'],?\s+(?:said|replied|asked|exclaimed|whispered)\s+{character_name}',
            rf'{character_name}\s+(?:said|replied|asked|exclaimed|whispered),?\s+["\'](.+?)["\']'
        ]
        
        quotes = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                quote = match.group(1).strip()
                if 10 < len(quote) < 200:  # Reasonable quote length
                    quotes.append(quote)
        
        # Add top quotes to character data
        character["quotes"] = quotes[:5]  # Keep top 5 quotes

def extract_themes_with_embeddings(chapters, nlp, st_model, batch_size=32):
    """
    Extract themes using semantic embeddings for better clustering and relevance.
    """
    print("Analyzing themes with semantic embeddings...")
    
    # Prepare text samples
    text_samples = []
    for chapter in chapters:
        # Use spaCy for sentence tokenization instead of NLTK
        doc = nlp(chapter[:300000])  # Limit to first 300K chars per chapter
        sentences = [sent.text for sent in doc.sents]
        
        # Filter for sentences that might contain themes (reasonable length, not dialogue)
        meaningful_sentences = [s for s in sentences 
                               if 40 < len(s) < 300 
                               and not s.startswith('"') 
                               and '?' not in s[:10]]
        
        # Sample sentences (for large chapters)
        if len(meaningful_sentences) > 200:
            sampled = []
            step = len(meaningful_sentences) // 200
            for i in range(0, len(meaningful_sentences), step):
                sampled.append(meaningful_sentences[i])
            meaningful_sentences = sampled
        
        text_samples.extend(meaningful_sentences)
        
    # Ensure we don't process too many samples (RTX 3090 memory optimization)
    if len(text_samples) > 1000:
        # Random sample from all sentences for diversity
        import random
        random.shuffle(text_samples)
        text_samples = text_samples[:1000]
    
    print(f"Encoding {len(text_samples)} sentences for theme analysis...")
    
    # Generate embeddings for sentences
    embeddings = st_model.encode(text_samples, batch_size=batch_size, show_progress_bar=True)
    
    # Cluster similar sentences
    clustering = DBSCAN(eps=0.3, min_samples=3).fit(embeddings)
    
    # Group sentences by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        if label >= 0:  # Ignore noise points (-1)
            clusters[label].append(text_samples[i])
    
    print(f"Found {len(clusters)} potential theme clusters")
    
    # Process each cluster to extract theme words
    theme_clusters = []
    
    for cluster_id, sentences in clusters.items():
        # Skip small clusters
        if len(sentences) < 3:
            continue
            
        # Process sentences to extract key phrases
        cluster_doc = nlp(" ".join(sentences[:10]))  # Process subset for efficiency
        
        # Extract noun chunks, adjectives, and important words
        key_phrases = []
        for chunk in cluster_doc.noun_chunks:
            if 3 < len(chunk.text) < 30 and not all(token.is_stop for token in chunk):
                key_phrases.append(chunk.text.lower())
        
        important_words = []
        for token in cluster_doc:
            if not token.is_stop and token.pos_ in ["NOUN", "ADJ", "VERB"] and len(token.text) > 3:
                important_words.append(token.lemma_.lower())
        
        # Count frequencies
        phrase_counts = Counter(key_phrases)
        word_counts = Counter(important_words)
        
        # Skip clusters without clear themes
        if not phrase_counts and not word_counts:
            continue
            
        # Get most common phrases and words
        top_phrases = [phrase for phrase, count in phrase_counts.most_common(5) if count > 1]
        top_words = [word for word, count in word_counts.most_common(10) if count > 2]
        
        # Determine theme name (use most common phrase or word combination)
        if top_phrases:
            theme_name = top_phrases[0]
        elif top_words:
            theme_name = top_words[0]
            # Try to combine adjective+noun if available
            adj_and_nouns = [(w, pos) for w, pos in [(word, nlp(word)[0].pos_) for word in top_words] 
                            if pos in ["ADJ", "NOUN"]]
            if len(adj_and_nouns) > 1:
                adj = next((w for w, pos in adj_and_nouns if pos == "ADJ"), None)
                noun = next((w for w, pos in adj_and_nouns if pos == "NOUN"), None)
                if adj and noun:
                    theme_name = f"{adj} {noun}"
        else:
            continue  # Skip if no clear theme
        
        # Get example sentences (best representatives of the theme)
        # Use embedding similarity to cluster centroid for better examples
        if len(sentences) > 2:
            cluster_indices = [i for i, label in enumerate(clustering.labels_) if label == cluster_id]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find sentences closest to centroid
            similarities = np.dot(cluster_embeddings, centroid) / (
                np.linalg.norm(cluster_embeddings, axis=1) * np.linalg.norm(centroid)
            )
            best_indices = np.argsort(-similarities)[:3]  # Top 3 similar sentences
            example_sentences = [sentences[cluster_indices[i]] for i in best_indices]
        else:
            example_sentences = sentences
        
        theme_clusters.append({
            "theme": theme_name,
            "keywords": top_words,
            "related_phrases": top_phrases,
            "examples": example_sentences,
            "strength": len(sentences)  # Number of sentences in cluster indicates theme strength
        })
    
    # Sort by theme strength
    theme_clusters.sort(key=lambda x: x["strength"], reverse=True)
    
    return theme_clusters[:20]  # Return top 20 themes

def extract_motifs_and_symbols(text, nlp, st_model, characters):
    """Extract recurring motifs and symbolic elements."""
    print("Identifying motifs and symbols...")
    
    # Split into chunks
    chunks = split_text_into_chunks(text, size=8000)
    
    # Process in batches to extract repeated imagery
    repeated_imagery = []
    processed_docs = list(nlp.pipe(chunks[:30], batch_size=16))  # Process subset for efficiency
    
    # Extract noun phrases that might represent motifs
    imagery_phrases = []
    for doc in processed_docs:
        # Look for vivid descriptions (adjective+noun combinations)
        for chunk in doc.noun_chunks:
            has_adj = any(token.pos_ == "ADJ" for token in chunk)
            if has_adj and 3 < len(chunk.text) < 25:
                imagery_phrases.append(chunk.text.lower())
        
        # Look for repeated objects, places, or elements of nature
        for ent in doc.ents:
            if ent.label_ in ["LOC", "FAC", "ORG", "PRODUCT"]:
                imagery_phrases.append(ent.text.lower())
    
    # Count frequency
    phrase_counts = Counter(imagery_phrases)
    
    # Filter for motifs (repeated imagery, not character names)
    character_names = set()
    for char in characters:
        character_names.add(char["name"].lower())
        for variant in char.get("variants", []):
            character_names.add(variant.lower())
    
    motifs = []
    for phrase, count in phrase_counts.most_common(50):
        if count >= 3 and phrase not in character_names and len(phrase) > 3:
            # Find example occurrences
            examples = []
            pattern = re.compile(r'[^.!?]*\b' + re.escape(phrase) + r'\b[^.!?]*[.!?]')
            for chunk in chunks:
                matches = pattern.findall(chunk.lower())
                for match in matches[:2]:  # Get up to 2 examples per chunk
                    if 20 < len(match) < 300:
                        examples.append(match.strip())
                if len(examples) >= 3:
                    break
            
            motifs.append({
                "motif": phrase,
                "occurrences": count,
                "examples": examples[:3]  # Up to 3 examples
            })
    
    # Try to categorize motifs
    categorized_motifs = {
        "nature": [],
        "objects": [],
        "locations": [],
        "color": [],
        "other": []
    }
    
    nature_words = ["sky", "moon", "sun", "water", "tree", "flower", "sea", "river", "mountain"]
    color_words = ["red", "blue", "green", "yellow", "white", "black", "gray", "purple"]
    location_words = ["house", "room", "garden", "building", "city", "town", "street"]
    
    for motif in motifs:
        phrase = motif["motif"]
        categorized = False
        
        for word in nature_words:
            if word in phrase:
                categorized_motifs["nature"].append(motif)
                categorized = True
                break
                
        if not categorized:
            for word in color_words:
                if word in phrase:
                    categorized_motifs["color"].append(motif)
                    categorized = True
                    break
                    
        if not categorized:
            for word in location_words:
                if word in phrase:
                    categorized_motifs["locations"].append(motif)
                    categorized = True
                    break
                    
        if not categorized:
            # Check if it's likely an object
            doc = nlp(phrase)
            if any(token.pos_ == "NOUN" for token in doc):
                categorized_motifs["objects"].append(motif)
            else:
                categorized_motifs["other"].append(motif)
    
    return {
        "all_motifs": motifs[:25],
        "categorized": categorized_motifs
    }

def analyze_character_relationships(text, characters, nlp):
    """Analyze relationships and interactions between characters."""
    if not characters:
        return []
    
    print(f"Analyzing relationships between {len(characters)} characters...")
    
    # Build a graph of character relationships
    graph = nx.Graph()
    
    # Create a mapping of character variants to canonical names
    character_map = {}
    for char in characters:
        canonical_name = char["name"]
        graph.add_node(canonical_name, mentions=char["mentions"])
        
        # Map all variants to the canonical name
        for variant in char.get("variants", [canonical_name]):
            character_map[variant.lower()] = canonical_name
    
    # Extract character co-occurrences using a window-based approach
    chunk_size = 3000
    chunks = split_text_into_chunks(text, size=chunk_size, overlap=200)
    
    # Process chunks to find co-occurring characters
    for chunk in chunks:
        chars_in_chunk = set()
        
        # Check which characters appear in this chunk
        for char_variant in character_map:
            if char_variant in chunk.lower():
                canonical = character_map[char_variant]
                chars_in_chunk.add(canonical)
        
        # Find dialogue interactions
        dialogue_pattern = r'["\'](.*?)["\']\s*,?\s*(?:said|replied|asked|answered|commented)\s+(\w+)'
        for match in re.finditer(dialogue_pattern, chunk):
            speaker = match.group(2).strip().lower()
            if speaker in character_map:
                canonical_speaker = character_map[speaker]
                # Look for characters mentioned in the dialogue
                dialogue = match.group(1).lower()
                for char_variant in character_map:
                    if char_variant in dialogue and character_map[char_variant] != canonical_speaker:
                        canonical_mentioned = character_map[char_variant]
                        # Add dialogue interaction
                        if graph.has_edge(canonical_speaker, canonical_mentioned):
                            graph[canonical_speaker][canonical_mentioned]['dialogue'] = graph[canonical_speaker][canonical_mentioned].get('dialogue', 0) + 1
                        else:
                            graph.add_edge(canonical_speaker, canonical_mentioned, dialogue=1)
        
        # Add co-occurrence edges
        char_list = list(chars_in_chunk)
        for i, char1 in enumerate(char_list):
            for char2 in char_list[i+1:]:
                if graph.has_edge(char1, char2):
                    graph[char1][char2]['weight'] = graph[char1][char2].get('weight', 0) + 1
                else:
                    graph.add_edge(char1, char2, weight=1)
    
    # Extract relationship types using NLP patterns
    for i, chunk in enumerate(chunks[:50]):  # Process subset for efficiency
        doc = nlp(chunk)
        for sent in doc.sents:
            char_pairs_in_sent = []
            
            # Look for character pairs in this sentence
            for char1 in graph.nodes():
                if char1.lower() in sent.text.lower():
                    for char2 in graph.neighbors(char1):
                        if char2.lower() in sent.text.lower() and char1 != char2:
                            char_pairs_in_sent.append((char1, char2))
            
            # If we found character pairs, look for relationship indicators
            for char1, char2 in char_pairs_in_sent:
                # Look for relationship verbs and nouns
                relationship_terms = []
                for token in sent:
                    # Relationship verbs (married, loves, hates, etc.)
                    if token.pos_ == "VERB" and token.lemma_ in ["marry", "love", "hate", "kill", "help", "betray"]:
                        relationship_terms.append(token.lemma_)
                    
                    # Relationship nouns (friend, enemy, brother, etc.)
                    if token.pos_ == "NOUN" and token.lemma_ in ["friend", "enemy", "brother", "sister", 
                                                               "father", "mother", "son", "daughter", 
                                                               "husband", "wife", "lover"]:
                        relationship_terms.append(token.lemma_)
                
                # If we found relationship terms, add them to the edge
                if relationship_terms:
                    if 'relationship_terms' not in graph[char1][char2]:
                        graph[char1][char2]['relationship_terms'] = []
                    graph[char1][char2]['relationship_terms'].extend(relationship_terms)
                    
                    # Also store the sentence as evidence
                    if 'evidence' not in graph[char1][char2]:
                        graph[char1][char2]['evidence'] = []
                    if len(sent.text) > 20:
                        graph[char1][char2]['evidence'].append(sent.text)
    
    # Extract relationships data
    relationships = []
    for char1, char2, data in graph.edges(data=True):
        weight = data.get('weight', 0)
        dialogue = data.get('dialogue', 0)
        relationship_terms = data.get('relationship_terms', [])
        evidence = data.get('evidence', [])
        
        # Only include significant relationships
        if weight > 1 or dialogue > 0:
            relationship = {
                "character1": char1,
                "character2": char2,
                "co_occurrences": weight,
                "dialogue_interactions": dialogue,
                "total_interactions": weight + dialogue
            }
            
            # Add relationship type if we found any
            if relationship_terms:
                term_counts = Counter(relationship_terms)
                relationship["relationship_indicators"] = [
                    {"term": term, "count": count} 
                    for term, count in term_counts.most_common()
                ]
            
            # Add evidence
            if evidence:
                relationship["evidence"] = evidence[:3]  # Top 3 pieces of evidence
            
            relationships.append(relationship)
    
    # Sort by total interactions
    relationships.sort(key=lambda x: x["total_interactions"], reverse=True)
    
    return relationships[:30]  # Return top 30 relationships

def extract_plotlines(text, characters, nlp, st_model):
    """Extract key plotlines and narrative arcs."""
    print("Extracting plotlines and narrative arcs...")
    
    # Get character names for reference
    character_names = set()
    for char in characters:
        character_names.add(char["name"].lower())
        for variant in char.get("variants", []):
            character_names.add(variant.lower())
    
    # Split text into narrative chunks
    chunks = split_text_into_chunks(text, size=5000, overlap=500)
    
    # Process chunks to extract plot events
    plot_events = []
    
    # Process a subset of chunks for efficiency
    for i, chunk in enumerate(chunks[:40]):  # Process first 40 chunks
        # Skip if chunk seems to be front/back matter
        if len(chunk) < 1000 or "copyright" in chunk.lower() or "contents" in chunk.lower():
            continue
            
        doc = nlp(chunk)
        
        # Extract sentences that might describe events
        for sent in doc.sents:
            # Skip short or dialogue sentences
            if len(sent.text) < 30 or sent.text.strip().startswith('"'):
                continue
                
            # Check if sentence contains a character
            has_character = any(char_name in sent.text.lower() for char_name in character_names)
            
            # Check if sentence has narrative structure (subject-verb-object)
            has_svo = False
            has_verb = False
            
            for token in sent:
                if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ != "PRON":
                    if any(token.pos_ == "VERB" for token in sent):
                        has_svo = True
                        break
                
                if token.pos_ == "VERB" and token.dep_ not in ["aux", "auxpass"]:
                    has_verb = True
            
            # Keep sentences with characters or narrative structure
            if (has_character and has_verb) or has_svo:
                # Clean the sentence
                clean_sent = re.sub(r'\s+', ' ', sent.text).strip()
                if 30 < len(clean_sent) < 200:
                    plot_events.append(clean_sent)
    
    # Limit number of events for processing
    if len(plot_events) > 100:
        plot_events = plot_events[:100]
    
    # Generate embeddings for plot events
    print(f"Analyzing {len(plot_events)} plot events...")
    event_embeddings = st_model.encode(plot_events, show_progress_bar=False)
    
    # Cluster events into narrative arcs
    clustering = DBSCAN(eps=0.35, min_samples=2).fit(event_embeddings)
    
    # Group events by cluster
    narrative_arcs = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        if label >= 0:  # Ignore noise
            narrative_arcs[label].append(plot_events[i])
    
    # Prepare plotline data
    plotlines = []
    
    for arc_id, events in narrative_arcs.items():
        # Skip small arcs
        if len(events) < 2:
            continue
            
        # Generate a descriptive name for the arc
        arc_text = " ".join(events)
        arc_doc = nlp(arc_text)
        
        # Extract key entities and actions
        key_entities = []
        for ent in arc_doc.ents:
            if ent.label_ in ["PERSON", "ORG", "LOC", "GPE"]:
                key_entities.append(ent.text)
        
        key_verbs = []
        for token in arc_doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "xcomp"] and not token.is_stop:
                key_verbs.append(token.lemma_)
        
        # Count frequencies
        entity_counts = Counter(key_entities)
        verb_counts = Counter(key_verbs)
        
        # Create plot arc name from most common entity and verb
        top_entities = [e for e, c in entity_counts.most_common(2)]
        top_verbs = [v for v, c in verb_counts.most_common(2)]
        
        if top_entities and top_verbs:
            arc_name = f"{top_entities[0]} {top_verbs[0]}"
            if len(top_entities) > 1:
                arc_name += f" with {top_entities[1]}"
        elif top_entities:
            arc_name = f"Events involving {top_entities[0]}"
        elif top_verbs:
            arc_name = f"Actions: {', '.join(top_verbs[:3])}"
        else:
            arc_name = f"Plot arc {arc_id+1}"
        
        # Add plotline to results
        plotlines.append({
            "plotline": arc_name,
            "events": events,
            "importance": len(events)
        })
    
    # Sort by importance
    plotlines.sort(key=lambda x: x["importance"], reverse=True)
    
    return plotlines[:15]  # Return top 15 plotlines

def analyze_text_for_rag(book_data, args):
    """Perform comprehensive literary analysis optimized for RAG operations."""
    # Configure GPU settings
    gpu_settings = optimize_gpu_settings()
    
    # Update batch size if optimal value was found
    batch_size = args.batch_size
    if gpu_settings.get("use_gpu") and "optimal_batch_size" in gpu_settings:
        batch_size = max(batch_size, gpu_settings["optimal_batch_size"])
        print(f"Using batch size: {batch_size}")
    
    # Load NLP models
    nlp, st_model = load_nlp_models(args.model, args.embedding_model, args.fp16)
    
    # Start total timer
    start_time = time.time()
    
    # Extract and process text
    full_text = preprocess_text_for_rag(book_data["full_text"], args.max_text_length)
    chapters = [preprocess_text_for_rag(chapter, args.max_text_length // len(book_data["chapters"])) 
                for chapter in book_data["chapters"]]
    
    # Process with our enhanced analysis functions
    print("\nExtracting characters...")
    characters = extract_characters_advanced(full_text, nlp, st_model, batch_size)
    
    print("\nAnalyzing themes...")
    themes = extract_themes_with_embeddings(chapters, nlp, st_model, batch_size)
    
    print("\nIdentifying motifs and symbols...")
    motifs_data = extract_motifs_and_symbols(full_text, nlp, st_model, characters)
    
    print("\nAnalyzing character relationships...")
    relationships = analyze_character_relationships(full_text, characters, nlp)
    
    print("\nExtracting plotlines...")
    plotlines = extract_plotlines(full_text, characters, nlp, st_model)
    
    # Analyze chapters (if more than one)
    chapter_analysis = []
    if len(chapters) > 1:
        print(f"\nAnalyzing {len(chapters)} individual chapters...")
        for i, (chapter, title) in enumerate(zip(chapters, book_data["chapter_titles"])):
            print(f"  Processing chapter {i+1}: {title}")
            
            # Extract chapter-specific characters
            chapter_chars = extract_characters_advanced(chapter, nlp, st_model, batch_size)
            
            # Extract chapter-specific themes (simplified)
            chapter_themes = extract_themes_with_embeddings([chapter], nlp, st_model, batch_size)
            
            # Summarize chapter
            chapter_summary = ""
            if len(chapter) > 100:
                # Get representative sentences
                doc = nlp(chapter)
                sentences = [sent.text for sent in doc.sents]
                
                if len(sentences) > 10:
                    # Use sentence embeddings to find most central sentences
                    sample_size = min(50, len(sentences))
                    sampled_sentences = sentences[:sample_size]
                    
                    # Generate embeddings
                    sent_embeddings = st_model.encode(sampled_sentences, show_progress_bar=False)
                    
                    # Find centroid
                    centroid = np.mean(sent_embeddings, axis=0)
                    
                    # Calculate sentence similarities to centroid
                    similarities = np.dot(sent_embeddings, centroid) / (
                        np.linalg.norm(sent_embeddings, axis=1) * np.linalg.norm(centroid)
                    )
                    
                    # Get top sentences
                    top_indices = np.argsort(-similarities)[:3]
                    summary_sentences = [sampled_sentences[i] for i in top_indices]
                    chapter_summary = " ".join(summary_sentences)
                else:
                    chapter_summary = " ".join(sentences[:3])
            
            chapter_analysis.append({
                "title": title,
                "characters": chapter_chars[:10],
                "themes": chapter_themes[:5],
                "summary": chapter_summary
            })
    
    # Create integrated analysis with RAG-friendly structure
    analysis = {
        "title": book_data["title"],
        "author": book_data["author"],
        "characters": characters,
        "themes": themes,
        "motifs": motifs_data["all_motifs"],
        "motifs_by_category": motifs_data["categorized"],
        "plotlines": plotlines,
        "relationships": relationships,
        "chapters": chapter_analysis,
        "statistics": {
            "word_count": len(full_text.split()),
            "character_count": len(characters),
            "chapter_count": len(chapters),
            "processing_time_seconds": time.time() - start_time
        },
        "metadata": {
            "analysis_version": "2.0",
            "spacy_model": args.model,
            "embedding_model": args.embedding_model,
            "analyzed_on": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    return analysis

def save_analysis_for_rag(analysis, book_path, output_dir):
    """Save analysis in RAG-optimized format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a clean filename base
    base_name = os.path.splitext(os.path.basename(book_path))[0]
    base_name = re.sub(r'[^\w\-\.]', '_', base_name)
    
    print(f"Saving RAG-optimized analysis to {output_dir}/")
    
    # Save complete analysis
    with open(f"{output_dir}/{base_name}_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Save RAG-optimized components in separate files
    
    # 1. Character profiles with context
    character_data = {
        "book_title": analysis["title"],
        "book_author": analysis["author"],
        "character_count": len(analysis["characters"]),
        "characters": analysis["characters"]
    }
    with open(f"{output_dir}/{base_name}_characters.json", 'w', encoding='utf-8') as f:
        json.dump(character_data, f, indent=2, ensure_ascii=False)
    
    # 2. Themes with examples
    theme_data = {
        "book_title": analysis["title"],
        "book_author": analysis["author"],
        "theme_count": len(analysis["themes"]),
        "themes": analysis["themes"]
    }
    with open(f"{output_dir}/{base_name}_themes.json", 'w', encoding='utf-8') as f:
        json.dump(theme_data, f, indent=2, ensure_ascii=False)
    
    # 3. Motifs and symbols
    motif_data = {
        "book_title": analysis["title"],
        "book_author": analysis["author"],
        "motif_count": len(analysis["motifs"]),
        "motifs": analysis["motifs"],
        "categories": analysis["motifs_by_category"]
    }
    with open(f"{output_dir}/{base_name}_motifs.json", 'w', encoding='utf-8') as f:
        json.dump(motif_data, f, indent=2, ensure_ascii=False)
    
    # 4. Character relationships
    relationship_data = {
        "book_title": analysis["title"],
        "book_author": analysis["author"],
        "relationship_count": len(analysis["relationships"]),
        "relationships": analysis["relationships"]
    }
    with open(f"{output_dir}/{base_name}_relationships.json", 'w', encoding='utf-8') as f:
        json.dump(relationship_data, f, indent=2, ensure_ascii=False)
    
    # 5. Plotlines/narrative arcs
    plotline_data = {
        "book_title": analysis["title"],
        "book_author": analysis["author"],
        "plotline_count": len(analysis["plotlines"]),
        "plotlines": analysis["plotlines"]
    }
    with open(f"{output_dir}/{base_name}_plotlines.json", 'w', encoding='utf-8') as f:
        json.dump(plotline_data, f, indent=2, ensure_ascii=False)
    
    # 6. Chapter summaries/analysis
    if analysis["chapters"]:
        chapter_data = {
            "book_title": analysis["title"],
            "book_author": analysis["author"],
            "chapter_count": len(analysis["chapters"]),
            "chapters": analysis["chapters"]
        }
        with open(f"{output_dir}/{base_name}_chapters.json", 'w', encoding='utf-8') as f:
            json.dump(chapter_data, f, indent=2, ensure_ascii=False)
    
    # 7. RAG context file (combined useful elements in a single file)
    rag_context = {
        "book": {
            "title": analysis["title"],
            "author": analysis["author"],
            "word_count": analysis["statistics"]["word_count"],
            "chapter_count": analysis["statistics"]["chapter_count"]
        },
        "main_characters": [
            {
                "name": char["name"],
                "mentions": char["mentions"],
                "quotes": char.get("quotes", [])[:2],  # Include sample quotes if available
                "context": char.get("contexts", [])[:1]  # Brief context
            } 
            for char in analysis["characters"][:10]  # Top 10 characters
        ],
        "key_relationships": [
            {
                "character1": rel["character1"],
                "character2": rel["character2"],
                "interactions": rel["total_interactions"],
                "evidence": rel.get("evidence", [""])[0]
            }
            for rel in analysis["relationships"][:10]  # Top 10 relationships
        ],
        "major_themes": [
            {
                "theme": theme["theme"],
                "keywords": theme.get("keywords", [])[:5],
                "example": theme.get("examples", [""])[0]
            }
            for theme in analysis["themes"][:10]  # Top 10 themes
        ],
        "important_motifs": [
            {
                "motif": motif["motif"],
                "occurrences": motif["occurrences"],
                "example": motif.get("examples", [""])[0]
            }
            for motif in analysis["motifs"][:10]  # Top 10 motifs
        ],
        "key_plotlines": [
            {
                "plotline": plot["plotline"],
                "events": plot["events"][:2]  # Sample events from plotline
            }
            for plot in analysis["plotlines"][:8]  # Top 8 plotlines
        ],
        "chapter_overview": [
            {
                "title": chap["title"],
                "summary": chap["summary"],
                "key_characters": [char["name"] for char in chap["characters"][:5]]
            }
            for chap in analysis["chapters"][:20]  # Up to 20 chapters
        ]
    }
    
    with open(f"{output_dir}/{base_name}_rag_context.json", 'w', encoding='utf-8') as f:
        json.dump(rag_context, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(os.listdir(output_dir))} analysis files")
    return rag_context

def main():
    args = parse_arguments()
    
    total_start_time = time.time()
    
    for epub_file in args.files:
        try:
            file_start_time = time.time()
            print(f"\n{'='*60}")
            print(f"ANALYZING: {epub_file}")
            print(f"{'='*60}")
            
            # Extract text content
            book_data = extract_text_from_epub(epub_file, by_chapter=args.by_chapter)
            
            print(f"Book: '{book_data['title']}' by {book_data['author']}")
            print(f"Text length: {len(book_data['full_text'])} characters")
            print(f"Chapters identified: {len(book_data['chapters'])}")
            
            # Analyze the text with our enhanced RAG-optimized functions
            analysis = analyze_text_for_rag(book_data, args)
            
            # Report findings
            print("\nANALYSIS SUMMARY:")
            print(f"- Found {len(analysis['characters'])} characters")
            print(f"- Identified {len(analysis['themes'])} themes")
            print(f"- Detected {len(analysis['motifs'])} motifs")
            print(f"- Mapped {len(analysis['relationships'])} character relationships")
            print(f"- Extracted {len(analysis['plotlines'])} plotlines")
            
            # Save RAG-optimized analysis
            save_analysis_for_rag(analysis, epub_file, args.output_dir)
            
            print(f"\nAnalysis for '{analysis['title']}' completed successfully.")
            print(f"Processing time: {time.time() - file_start_time:.2f} seconds")
            
        except Exception as e:
            print(f"\nERROR processing {epub_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTotal processing time: {time.time() - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()