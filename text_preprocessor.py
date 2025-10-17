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
        self.nlp = spacy.load(lang_model)
    
    def clean_text(self, raw_text: str) -> str:
        """Clean text using SpaCy by removing punctuation, stop words, and numbers."""
        doc = self.nlp(raw_text.replace("\\n", "\n"))
        
        cleaned_lines = []
        for sent in doc.sents:
            tokens = [
                tok.text.lower().translate(str.maketrans('', '', string.punctuation))
                for tok in sent
                if not (tok.is_space or tok.is_stop or tok.is_punct or tok.like_num)
            ]
            
            if line := " ".join(t for t in tokens if t):
                cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)
    
    def clean_and_export_files(self, 
                             texts_dir: str = "texts",
                             source_patterns: Optional[List[str]] = None,
                             dir_prefix: str = "llm") -> None:
        """Clean and export text files from specified directories."""
        source_patterns = source_patterns or [f"{dir_prefix}_*/*.txt"]
        
        cleaned_dirs = {
            "complex": os.path.join(texts_dir, f"cleaned_{dir_prefix}_complex"),
            "vague": os.path.join(texts_dir, f"cleaned_{dir_prefix}_vague")
        }
        
        for cleaned_dir in cleaned_dirs.values():
            os.makedirs(cleaned_dir, exist_ok=True)
        
        # Find and filter files
        files_to_process = []
        for pattern in source_patterns:
            files_to_process.extend(glob.glob(os.path.join(texts_dir, pattern)))
        
        files_to_process = [
            path for path in files_to_process
            if not (path.endswith('_cleaned.txt') or 'emo_edges_' in path or 'edges_' in path)
        ]
        
        print(f"Found {len(files_to_process)} files to clean...")
        
        with tqdm(total=len(files_to_process), desc="Cleaning texts") as pbar:
            for path in files_to_process:
                filename = os.path.basename(path)
                pbar.set_description(f"Cleaning {filename}")
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        cleaned = self.clean_text(f.read())
                    
                    # Determine output directory
                    path_lower = path.lower()
                    if 'complex' in path_lower:
                        out_dir = cleaned_dirs["complex"]
                    elif 'vague' in path_lower:
                        out_dir = cleaned_dirs["vague"]
                    else:
                        pbar.update(1)
                        continue
                    
                    out_path = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_cleaned.txt")
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned)
                
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                
                pbar.update(1)
        
        print(f"\nText cleaning completed!")
        print(f"  - Files processed: {len(files_to_process)}")
        print(f"  - Cleaned files saved in: {list(cleaned_dirs.values())}")
    
    def clean_single_file(self, input_path: str, output_path: str) -> bool:
        """Clean a single text file."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                cleaned_text = self.clean_text(f.read())
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            return True
            
        except Exception as e:
            print(f"Error cleaning {input_path}: {e}")
            return False