"""
Text generation using LM Studio local API.
Refactored for generic LLM support with configurable paths.
"""

import requests
import os
from tqdm import tqdm
from typing import Dict, List, Optional


class LLMGenerator:
    """Handles text generation using LM Studio local API."""
    
    def __init__(self, model: str = "local-model", 
                 base_url: str = "http://localhost:1234/v1/chat/completions"):
        self.base_url = base_url
        self.model = model
        
    def generate_texts(self, 
                      prompts: Dict[str, str], 
                      temperatures: List[float],
                      n_completions: int = 5,
                      system_instruction: Optional[str] = None,
                      texts_dir: str = "texts",
                      dir_prefix: str = "llm") -> None:
        """Generate texts for all prompt-temperature combinations."""
        
        system_instruction = system_instruction or (
            "Please read the following prompt carefully and adopt the perspective of a writer. "
            "Your task is to continue the story in any direction you choose. "
            "The story should be between 250 and 350 words."
        )
        
        os.makedirs(texts_dir, exist_ok=True)
        total_ops = len(prompts) * len(temperatures)
        print(f"Starting generation of {total_ops} text sets...")
        
        with tqdm(total=total_ops, desc="Generating texts") as pbar:
            for prompt_name, prompt_text in prompts.items():
                out_dir = os.path.join(texts_dir, f"{dir_prefix}_{prompt_name}")
                os.makedirs(out_dir, exist_ok=True)
                
                for temp in temperatures:
                    pbar.set_description(f"Processing {prompt_name} T={temp}")
                    completions = self._generate_completions(
                        prompt_text, system_instruction, temp, n_completions
                    )
                    
                    file_path = os.path.join(out_dir, f"{prompt_name}_{temp}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("\n\n".join(completions))
                    
                    pbar.update(1)
        
        print(f"\nGeneration completed!")
        print(f"  - Processed prompts: {len(prompts)}")
        print(f"  - Temperatures per prompt: {len(temperatures)}")
        print(f"  - Completions per temperature: {n_completions}")
        print(f"  - Total texts generated: {len(prompts) * len(temperatures) * n_completions}")
    
    def _generate_completions(self, prompt_text: str, system_instruction: str,
                            temperature: float, n_completions: int) -> List[str]:
        """Generate completions for a specific prompt and temperature."""
        completions = []
        
        for i in range(n_completions):
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt_text},
                    ],
                    "temperature": temperature,
                }
                
                response = requests.post(self.base_url, json=payload, timeout=1200)
                response.raise_for_status()
                completion = response.json()['choices'][0]['message']['content'].strip()
                completions.append(completion)
                
            except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                print(f"Error generating completion {i+1}/{n_completions}: {e}")
                break
        
        return completions