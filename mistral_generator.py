"""
Text generation using Mistral AI API.
Extracted from Hakan.ipynb for better modularity.
"""

from mistralai import Mistral
import os
from tqdm import tqdm
from typing import Dict, List, Optional


class MistralGenerator:
    """Handles text generation using Mistral AI API."""
    
    def __init__(self, api_key: str, model: str = "mistral-small-latest"):
        """
        Initialize the Mistral generator.
        
        Args:
            api_key: Mistral API key
            model: Model name to use
        """
        self.client = Mistral(api_key=api_key)
        self.model = model
        
    def generate_texts(self, 
                      prompts: Dict[str, str], 
                      temperatures: List[float],
                      n_completions: int = 5,
                      system_instruction: Optional[str] = None,
                      texts_dir: str = "texts") -> None:
        """
        Generate texts for all prompt-temperature combinations.
        
        Args:
            prompts: Dictionary with prompt names and texts
            temperatures: List of temperature values
            n_completions: Number of completions per temperature
            system_instruction: System instruction for the model
            texts_dir: Output directory for generated texts
        """
        if system_instruction is None:
            system_instruction = (
                "Please read the following prompt carefully and adopt the perspective of a writer. "
                "Your task is to continue the story in any direction you choose. "
                "The story should be between 250 and 350 words."
            )
        
        # Create main texts directory
        os.makedirs(texts_dir, exist_ok=True)
        
        # Calculate total operations for progress bar
        total_operations = len(prompts) * len(temperatures)
        print(f"Starting generation of {total_operations} text sets...")
        
        # Unified progress bar
        with tqdm(total=total_operations, desc="Generating texts") as pbar:
            for prompt_name, prompt_text in prompts.items():
                # Create directory with the prompt name inside texts folder
                out_dir = os.path.join(texts_dir, f"mistral_{prompt_name}")
                os.makedirs(out_dir, exist_ok=True)
                
                for temp in temperatures:
                    # Update progress bar description
                    pbar.set_description(f"Processing {prompt_name} T={temp}")
                    
                    completions = self._generate_completions(
                        prompt_text, system_instruction, temp, n_completions
                    )
                    
                    # Write texts to file separated by two blank lines
                    file_path = os.path.join(out_dir, f"{prompt_name}_{temp}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("\n\n".join(completions))
                    
                    # Update progress bar
                    pbar.update(1)
        
        print(f"\nGeneration completed!")
        print(f"  - Processed prompts: {len(prompts)}")
        print(f"  - Temperatures per prompt: {len(temperatures)}")
        print(f"  - Completions per temperature: {n_completions}")
        print(f"  - Total texts generated: {len(prompts) * len(temperatures) * n_completions}")
    
    def _generate_completions(self, 
                            prompt_text: str, 
                            system_instruction: str,
                            temperature: float,
                            n_completions: int) -> List[str]:
        """
        Generate completions for a specific prompt and temperature.
        
        Args:
            prompt_text: The prompt text
            system_instruction: System instruction
            temperature: Temperature value
            n_completions: Number of completions to generate
            
        Returns:
            List of generated completions
        """
        completions = []
        
        while len(completions) < n_completions:
            batch_size = min(10, n_completions - len(completions))
            
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt_text},
                    ],
                    temperature=temperature,
                    n=batch_size,
                )
                
                batch = [choice.message.content.strip() for choice in response.choices]
                completions.extend(batch)
                
            except Exception as e:
                print(f"Error generating completions: {e}")
                break
        
        return completions