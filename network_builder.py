"""
Network construction using EmoAtlas.
Extracted from Hakan.ipynb for better modularity.
"""

import os
import glob
import nltk
from emoatlas import EmoScores
from tqdm import tqdm
from typing import Dict, List, Optional


class NetworkBuilder:
    """Handles semantic network construction using EmoAtlas."""
    
    def __init__(self):
        """Initialize the network builder."""
        # Download required NLTK data
        try:
            nltk.download('wordnet', quiet=True)
        except:
            pass
        
        # Initialize EmoAtlas
        self.emos = EmoScores()
    
    def build_networks_from_texts(self, 
                                 texts_dir: str = "texts",
                                 source_dirs: Optional[List[str]] = None) -> None:
        """
        Build emotional networks from text files using EmoAtlas.
        
        Args:
            texts_dir: Base directory containing text files
            source_dirs: List of source directories to process
        """
        if source_dirs is None:
            source_dirs = [
                os.path.join(texts_dir, 'mistral_prompt_complex'),
                os.path.join(texts_dir, 'mistral_prompt_vague')
            ]
        
        # Count total files for progress bar
        total_files = 0
        for in_dir in source_dirs:
            if os.path.exists(in_dir):
                pattern = os.path.join(in_dir, '*.txt')
                files = [f for f in glob.glob(pattern) if not f.endswith('_cleaned.txt')]
                total_files += len(files)
        
        print(f"Found {total_files} files to process with EmoAtlas...")
        
        # Process files with progress bar
        with tqdm(total=total_files, desc="Creating emotional networks") as pbar:
            for in_dir in source_dirs:
                if not os.path.exists(in_dir):
                    print(f"Warning: Directory {in_dir} not found")
                    continue
                
                # Extract the prompt type: 'complex' or 'vague'
                prompt_type = self._extract_prompt_type(in_dir)
                
                # Create the output directory for emotional edge lists
                out_dir = f'emo_edges_{prompt_type}'
                os.makedirs(out_dir, exist_ok=True)
                
                # Process each original text file
                pattern = os.path.join(in_dir, '*.txt')
                for path in glob.glob(pattern):
                    filename = os.path.basename(path)
                    
                    # Skip already cleaned files
                    if filename.endswith('_cleaned.txt'):
                        continue
                    
                    # Update progress bar description
                    pbar.set_description(f"Processing {filename}")
                    
                    # Extract temperature from filename
                    temp = self._extract_temperature(filename)
                    if temp is None:
                        print(f"Skip {filename}: unable to extract temperature")
                        pbar.update(1)
                        continue
                    
                    # Process the file
                    success = self._process_single_file(path, out_dir, prompt_type, temp)
                    if not success:
                        print(f"Failed to process {filename}")
                    
                    # Update progress bar
                    pbar.update(1)
        
        print(f"\nEmoAtlas network creation completed!")
        print(f"  - Edge lists saved in: emo_edges_complex/ and emo_edges_vague/")
    
    def _extract_prompt_type(self, directory_path: str) -> str:
        """
        Extract prompt type from directory path.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Prompt type ('complex' or 'vague')
        """
        if 'complex' in directory_path.lower():
            return 'complex'
        elif 'vague' in directory_path.lower():
            return 'vague'
        else:
            return 'unknown'
    
    def _extract_temperature(self, filename: str) -> Optional[float]:
        """
        Extract temperature value from filename.
        
        Args:
            filename: The filename to process
            
        Returns:
            Temperature value or None if not found
        """
        # Extract the temperature value from the filename: prompt_[type]_[temperature].txt
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        
        if len(parts) >= 2:
            try:
                return float(parts[-1])  # the last part is the temperature value
            except ValueError:
                return None
        
        return None
    
    def _process_single_file(self, 
                           input_path: str, 
                           output_dir: str, 
                           prompt_type: str, 
                           temperature: float) -> bool:
        """
        Process a single text file and create emotional network.
        
        Args:
            input_path: Path to input text file
            output_dir: Output directory for edge lists
            prompt_type: Type of prompt ('complex' or 'vague')
            temperature: Temperature value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the original text
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Building the emotional network using EmoAtlas
            try:
                # FormaMentis creates a semantic network based on emotional relationships
                network = self.emos.formamentis_network(text)
                
                # Verify that the network actually contains edges
                if not hasattr(network, 'edges') or not network.edges:
                    print(f"Warning: No edges created for {os.path.basename(input_path)}")
                    return False
                
                # Export the edge list
                out_path = os.path.join(output_dir, f'emo_edge_list_{prompt_type}_{temperature}.txt')
                with open(out_path, 'w', encoding='utf-8') as outf:
                    for u, v in network.edges:
                        outf.write(f"{u}\t{v}\n")
                
                return True
                
            except Exception as e:
                print(f"Error in network generation for {os.path.basename(input_path)}: {str(e)}")
                return False
        
        except Exception as e:
            print(f"Error in reading {input_path}: {str(e)}")
            return False
    
    def build_single_network(self, text: str) -> Optional[object]:
        """
        Build a single emotional network from text.
        
        Args:
            text: Input text
            
        Returns:
            Network object or None if failed
        """
        try:
            return self.emos.formamentis_network(text)
        except Exception as e:
            print(f"Error building network: {e}")
            return None
    
    def save_network_edges(self, network: object, output_path: str) -> bool:
        """
        Save network edges to file.
        
        Args:
            network: Network object
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for u, v in network.edges:
                    f.write(f"{u}\t{v}\n")
            
            return True
            
        except Exception as e:
            print(f"Error saving network edges: {e}")
            return False
