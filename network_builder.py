"""
Network construction using EmoAtlas.
Extracted from Hakan.ipynb for better modularity.
"""

import os
import glob
import nltk
from emoatlas import EmoScores
from tqdm import tqdm
from typing import List, Optional


class NetworkBuilder:
    """Handles semantic network construction using EmoAtlas."""
    
    def __init__(self):
        try:
            nltk.download('wordnet', quiet=True)
        except:
            pass
        self.emos = EmoScores()
    
    def build_networks_from_texts(self, 
                                 texts_dir: str = "texts",
                                 source_dirs: Optional[List[str]] = None,
                                 dir_prefix: str = "llm") -> None:
        """Build emotional networks from text files using EmoAtlas."""
        source_dirs = source_dirs or [
            os.path.join(texts_dir, f'{dir_prefix}_complex'),
            os.path.join(texts_dir, f'{dir_prefix}_vague')
        ]
        
        # Count total files
        total_files = sum(
            len([f for f in glob.glob(os.path.join(d, '*.txt')) if not f.endswith('_cleaned.txt')])
            for d in source_dirs if os.path.exists(d)
        )
        
        print(f"Found {total_files} files to process with EmoAtlas...")
        
        with tqdm(total=total_files, desc="Creating emotional networks") as pbar:
            for in_dir in source_dirs:
                if not os.path.exists(in_dir):
                    print(f"Warning: Directory {in_dir} not found")
                    continue
                
                prompt_type = self._extract_prompt_type(in_dir)
                out_dir = f'emo_edges_{prompt_type}'
                os.makedirs(out_dir, exist_ok=True)
                
                for path in glob.glob(os.path.join(in_dir, '*.txt')):
                    filename = os.path.basename(path)
                    
                    if filename.endswith('_cleaned.txt'):
                        continue
                    
                    pbar.set_description(f"Processing {filename}")
                    
                    if (temp := self._extract_temperature(filename)) is None:
                        print(f"Skip {filename}: unable to extract temperature")
                        pbar.update(1)
                        continue
                    
                    if not self._process_single_file(path, out_dir, prompt_type, temp):
                        print(f"Failed to process {filename}")
                    
                    pbar.update(1)
        
        print(f"\nEmoAtlas network creation completed!")
        print(f"  - Edge lists saved in: emo_edges_complex/ and emo_edges_vague/")
    
    def _extract_prompt_type(self, directory_path: str) -> str:
        """Extract prompt type from directory path."""
        lower = directory_path.lower()
        return 'complex' if 'complex' in lower else 'vague' if 'vague' in lower else 'unknown'
    
    def _extract_temperature(self, filename: str) -> Optional[float]:
        """Extract temperature value from filename."""
        parts = os.path.splitext(filename)[0].split('_')
        if len(parts) >= 2:
            try:
                return float(parts[-1])
            except ValueError:
                pass
        return None
    
    def _process_single_file(self, input_path: str, output_dir: str, 
                           prompt_type: str, temperature: float) -> bool:
        """Process a single text file and create emotional network."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            network = self.emos.formamentis_network(text)
            
            if not hasattr(network, 'edges') or not network.edges:
                print(f"Warning: No edges created for {os.path.basename(input_path)}")
                return False
            
            out_path = os.path.join(output_dir, f'emo_edge_list_{prompt_type}_{temperature}.txt')
            with open(out_path, 'w', encoding='utf-8') as outf:
                for u, v in network.edges:
                    outf.write(f"{u}\t{v}\n")
            
            return True
            
        except Exception as e:
            print(f"Error processing {os.path.basename(input_path)}: {e}")
            return False
    
    def build_single_network(self, text: str) -> Optional[object]:
        """Build a single emotional network from text."""
        try:
            return self.emos.formamentis_network(text)
        except Exception as e:
            print(f"Error building network: {e}")
            return None
    
    def save_network_edges(self, network: object, output_path: str) -> bool:
        """Save network edges to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for u, v in network.edges:
                    f.write(f"{u}\t{v}\n")
            
            return True
            
        except Exception as e:
            print(f"Error saving network edges: {e}")
            return False