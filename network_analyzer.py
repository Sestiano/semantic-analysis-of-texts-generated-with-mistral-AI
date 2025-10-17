"""
Network analysis and metrics calculation.
Extracted from Hakan.ipynb for better modularity.
"""

import pandas as pd
import numpy as np
import networkx as nx
import os
import glob
import re
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional


class NetworkAnalyzer:
    """Handles network analysis and metrics calculation."""
    
    def analyze_edge_list(self, file_path: str) -> Optional[Dict]:
        """Analyze an edge-list file and calculate network metrics."""
        try:
            G = nx.Graph()
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        G.add_edge(parts[0], parts[1])
            
            return self._calculate_metrics(G, file_path)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def _calculate_metrics(self, G: nx.Graph, file_path: str) -> Dict:
        """Calculate comprehensive network metrics."""
        n, m = G.number_of_nodes(), G.number_of_edges()
        degrees = [d for _, d in G.degree()]
        is_connected = nx.is_connected(G) and n > 1
        
        # Basic metrics
        density = (2 * m) / (n * (n - 1)) if n > 1 else 0
        avg_clustering = nx.average_clustering(G) if n > 0 else 0
        
        # Path length
        path_len = float('nan')
        if is_connected:
            try:
                path_len = nx.average_shortest_path_length(G)
            except Exception as e:
                print(f"Warning: Could not calculate path length for {os.path.basename(file_path)}: {e}")
        
        # Degree metrics
        avg_degree = sum(degrees) / n if n > 0 else 0
        max_degree = max(degrees) if degrees else 0
        std_degree = np.std(degrees) if degrees else 0
        
        return {
            'id': os.path.splitext(os.path.basename(file_path))[0],
            'prompt_type': self._extract_prompt_type(file_path),
            'temperature': self._extract_temperature(file_path),
            'nodes': n,
            'edges': m,
            'density': density,
            'clustering': avg_clustering,
            'path_len': path_len,
            'avg_deg': avg_degree,
            'max_deg': max_degree,
            'std_deg': std_degree,
            'diameter': nx.diameter(G) if is_connected else float('nan'),
            'transitivity': nx.transitivity(G) if n > 0 else 0,
            'file_path': file_path
        }
    
    def _extract_prompt_type(self, file_path: str) -> str:
        """Extract prompt type from file path."""
        lower = file_path.lower()
        return 'complex' if 'complex' in lower else 'vague' if 'vague' in lower else 'other'
    
    def _extract_temperature(self, file_path: str) -> float:
        """Extract temperature from filename."""
        temp_match = re.search(r'_(\d+(?:\.\d+)?)$', os.path.splitext(os.path.basename(file_path))[0])
        return float(temp_match.group(1)) if temp_match else np.nan
    
    def analyze_all_networks(self, 
                           edge_dirs: Optional[List[str]] = None,
                           output_file: str = "results/network_metrics.csv") -> pd.DataFrame:
        """Analyze all network files and return results DataFrame."""
        edge_dirs = edge_dirs or ['emo_edges_complex', 'emo_edges_vague']
        
        # Collect edge-list files
        files = []
        for d in edge_dirs:
            if Path(d).is_dir():
                found = glob.glob(os.path.join(d, '*edge_list_*.txt'))
                print(f"Found {len(found)} files in {d}")
                files.extend(found)
        
        if not files:
            raise ValueError('No edge-list files found.')
        
        # Analyze each file
        results = []
        print(f"\nAnalyzing {len(files)} files...")
        
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for fp in files:
                pbar.set_description(f"Processing {os.path.basename(fp)}")
                if metrics := self.analyze_edge_list(fp):
                    results.append(metrics)
                pbar.update(1)
        
        if not results:
            raise ValueError('No results to display.')
        
        # Create DataFrame and save
        df = pd.DataFrame(results)
        Path(os.path.dirname(output_file)).mkdir(exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f'Metrics saved to {output_file}')
        
        return df
    
    def calculate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for network metrics."""
        summary = {}
        metric_cols = ['nodes', 'edges', 'density', 'clustering', 'avg_deg']
        
        if 'prompt_type' in df.columns:
            summary['by_prompt_type'] = df.groupby('prompt_type')[metric_cols].agg(['mean', 'std', 'min', 'max']).round(3)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            summary['correlation_matrix'] = df[numeric_cols].corr().round(3)
        
        if 'temperature' in df.columns:
            summary['by_temperature'] = df.groupby('temperature')[metric_cols].mean().round(3)
        
        return summary
    
    def compare_prompt_types(self, df: pd.DataFrame) -> Dict:
        """Compare metrics between different prompt types."""
        comparison = {}
        
        if 'prompt_type' in df.columns and len(df['prompt_type'].unique()) >= 2:
            for metric in ['nodes', 'edges', 'density', 'avg_deg', 'clustering']:
                if metric in df.columns:
                    type_means = df.groupby('prompt_type')[metric].mean()
                    comparison[metric] = {
                        'means': type_means.to_dict(),
                        'difference': type_means.diff().dropna().to_dict()
                    }
        
        return comparison
    
    def analyze_single_network(self, file_path: str) -> Optional[Dict]:
        """Analyze a single network file."""
        return self.analyze_edge_list(file_path)
    
    def get_network_overview(self, df: pd.DataFrame) -> str:
        """Get a text overview of the network analysis results."""
        overview = ["=== NETWORK ANALYSIS OVERVIEW ===\n", f"Total networks analyzed: {len(df)}"]
        
        if 'prompt_type' in df.columns:
            overview.append(f"Networks by prompt type: {df['prompt_type'].value_counts().to_dict()}")
        
        # Metric ranges
        metric_cols = ['nodes', 'edges', 'density', 'clustering', 'avg_deg']
        available = [col for col in metric_cols if col in df.columns]
        
        if available:
            overview.append("\nMetric ranges:")
            for col in available:
                overview.append(f"  {col}: {df[col].min():.3f} - {df[col].max():.3f} (mean: {df[col].mean():.3f})")
        
        return "\n".join(overview)