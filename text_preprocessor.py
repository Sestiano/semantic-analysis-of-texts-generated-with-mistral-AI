"""
Text preprocessing using SpaCy.
Extracted from Hakan.ipynb for better modularity.
"""

import spacy
import glob
import string
import os
from tqdm import tqdm
from typing import List, Optional


class TextPreprocessor:
    """Handles text cleaning and preprocessing using SpaCy."""
    
    def __init__(self, lang_model: str = 'en_core_web_lg'):
        """
        Initialize the text preprocessor.
        
        Args:
            lang_model: SpaCy language model to use
        """
        self.lang_model = lang_model
        self.nlp = spacy.load(lang_model)
    
    def clean_text(self, raw_text: str) -> str:
        """
        Clean a text using SpaCy by removing punctuation, stop words, and numbers.
        
        Args:
            raw_text: The text to clean
            
        Returns:
            The cleaned text with one line per sentence
        """
        # Normalize newline characters
        raw_text = raw_text.replace("\\n", "\n")
        
        # Process the text with SpaCy
        doc = self.nlp(raw_text)
        
        # Process each sentence separately
        cleaned_lines = []
        for sent in doc.sents:
            # Extract only relevant tokens and remove special characters
            tokens = [
                tok.text.lower().translate(str.maketrans('', '', string.punctuation))
                for tok in sent
                if not tok.is_space      # remove spaces
                and not tok.is_stop      # remove stop words
                and not tok.is_punct     # remove punctuation
                and not tok.like_num     # remove numbers
            ]
            
            # Join tokens in a line
            line = " ".join(t for t in tokens if t)
            if line:  # add only non-empty lines
                cleaned_lines.append(line)
        
        # Return the cleaned text
        return "\n".join(cleaned_lines)
    
    def clean_and_export_files(self, 
                             texts_dir: str = "texts",
                             source_patterns: Optional[List[str]] = None,
                             dir_prefix: str = "llm") -> None:
        """
        Clean and export text files from specified directories.
        
        Args:
            texts_dir: Base directory containing text files
            source_patterns: List of glob patterns to match source files
            dir_prefix: Prefix for source and cleaned directories (default: "llm")
        """
        if source_patterns is None:
            source_patterns = [f"{dir_prefix}_*/*.txt"]
        
        # Create cleaned directories directly in texts folder
        cleaned_dirs = {
            "complex": os.path.join(texts_dir, f"cleaned_{dir_prefix}_complex"),
            "vague": os.path.join(texts_dir, f"cleaned_{dir_prefix}_vague")
        }
        
        # Create cleaned directories
        for cleaned_dir in cleaned_dirs.values():
            os.makedirs(cleaned_dir, exist_ok=True)
        
        # Find all text files to process
        files_to_process = []
        for pattern in source_patterns:
            search_pattern = os.path.join(texts_dir, pattern)
            found_files = glob.glob(search_pattern)
            files_to_process.extend(found_files)
        
        # Filter out already processed files
        files_to_process = [
            path for path in files_to_process
            if not (path.endswith('_cleaned.txt') or
                    'emo_edges_' in path or 'edges_' in path)
        ]
        
        print(f"Found {len(files_to_process)} files to clean...")
        
        # Process files with progress bar
        with tqdm(total=len(files_to_process), desc="Cleaning texts") as pbar:
            for path in files_to_process:
                # Update progress bar description
                filename = os.path.basename(path)
                pbar.set_description(f"Cleaning {filename}")
                
                try:
                    # Read the content of the original file
                    with open(path, 'r', encoding='utf-8') as f:
                        raw = f.read()
                    
                    # Apply text cleaning
                    cleaned = self.clean_text(raw)
                    
                    # Determine output directory based on source
                    if 'complex' in path.lower():
                        out_dir = cleaned_dirs["complex"]
                    elif 'vague' in path.lower():
                        out_dir = cleaned_dirs["vague"]
                    else:
                        pbar.update(1)
                        continue  # Skip if not matching expected pattern
                    
                    # Save the cleaned text in the appropriate cleaned directory
                    out_path = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_cleaned.txt")
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned)
                
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                
                # Update progress bar
                pbar.update(1)
        
        print(f"\nText cleaning completed!")
        print(f"  - Files processed: {len(files_to_process)}")
        print(f"  - Cleaned files saved in: {list(cleaned_dirs.values())}")
    
    def clean_single_file(self, input_path: str, output_path: str) -> bool:
        """
        Clean a single text file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            cleaned_text = self.clean_text(raw_text)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            return True
            
        except Exception as e:
            print(f"Error cleaning {input_path}: {e}")
            return False