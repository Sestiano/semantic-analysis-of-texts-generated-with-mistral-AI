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
from typing import Dict, List, Optional, Tuple


class NetworkAnalyzer:
    """Handles network analysis and metrics calculation."""
    
    def __init__(self):
        """Initialize the network analyzer."""
        pass
    
    def analyze_edge_list(self, file_path: str) -> Optional[Dict]:
        """
        Analyze an edge-list file and calculate network metrics.
        
        Args:
            file_path: Path to the edge-list file
            
        Returns:
            Dictionary with network metrics or None if failed
        """
        try:
            # Loading edges and building the graph in a single pass
            G = nx.Graph()
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        G.add_edge(parts[0], parts[1])
            
            # Calculate network metrics
            return self._calculate_metrics(G, file_path)
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def _calculate_metrics(self, G: nx.Graph, file_path: str) -> Dict:
        """
        Calculate comprehensive network metrics.
        
        Args:
            G: NetworkX graph
            file_path: Original file path for metadata
            
        Returns:
            Dictionary with calculated metrics
        """
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        # Basic metrics
        density = (2 * m) / (n * (n - 1)) if n > 1 else 0
        avg_clustering = nx.average_clustering(G) if n > 0 else 0
        
        # Path length (only if graph is connected)
        path_len = float('nan')
        if nx.is_connected(G) and n > 1:
            try:
                path_len = nx.average_shortest_path_length(G)
            except Exception as e:
                print(f"Warning: Could not calculate path length for {os.path.basename(file_path)}: {e}")
        
        # Degree-based metrics
        degrees = [d for _, d in G.degree()]
        avg_degree = sum(degrees) / n if n > 0 else 0
        max_degree = max(degrees) if degrees else 0
        std_degree = np.std(degrees) if degrees else 0
        
        # Additional metrics
        diameter = nx.diameter(G) if nx.is_connected(G) and n > 1 else float('nan')
        transitivity = nx.transitivity(G) if n > 0 else 0
        
        # Extract metadata from file path
        prompt_type = self._extract_prompt_type(file_path)
        temperature = self._extract_temperature(file_path)
        
        return {
            'id': os.path.splitext(os.path.basename(file_path))[0],
            'prompt_type': prompt_type,
            'temperature': temperature,
            'nodes': n,
            'edges': m,
            'density': density,
            'clustering': avg_clustering,
            'path_len': path_len,
            'avg_deg': avg_degree,
            'max_deg': max_degree,
            'std_deg': std_degree,
            'diameter': diameter,
            'transitivity': transitivity,
            'file_path': file_path
        }
    
    def _extract_prompt_type(self, file_path: str) -> str:
        """Extract prompt type from file path."""
        lower = file_path.lower()
        if 'complex' in lower:
            return 'complex'
        elif 'vague' in lower:
            return 'vague'
        return 'other'
    
    def _extract_temperature(self, file_path: str) -> float:
        """Extract temperature from filename."""
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        temp_match = re.search(r'_(\d+(?:\.\d+)?)$', file_name)
        return float(temp_match.group(1)) if temp_match else np.nan
    
    def analyze_all_networks(self, 
                           edge_dirs: Optional[List[str]] = None,
                           output_file: str = "results/network_metrics.csv") -> pd.DataFrame:
        """
        Analyze all network files and return results DataFrame.
        
        Args:
            edge_dirs: List of directories containing edge-list files
            output_file: Path to save the results CSV
            
        Returns:
            DataFrame with network metrics
        """
        if edge_dirs is None:
            edge_dirs = ['emo_edges_complex', 'emo_edges_vague']
        
        # Collection of edge-list files
        files = []
        for d in edge_dirs:
            if Path(d).is_dir():
                found = glob.glob(os.path.join(d, '*edge_list_*.txt'))
                print(f"Found {len(found)} files in {d}")
                files.extend(found)
        
        if not files:
            raise ValueError('No edge-list files found.')
        
        # Analysis of each file with progress bar
        results = []
        print(f"\nAnalyzing {len(files)} files...")
        
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for fp in files:
                pbar.set_description(f"Processing {os.path.basename(fp)}")
                metrics = self.analyze_edge_list(fp)
                if metrics:
                    results.append(metrics)
                pbar.update(1)
        
        if not results:
            raise ValueError('No results to display.')
        
        # Creating DataFrame with results
        df = pd.DataFrame(results)
        
        # Save metrics to CSV
        Path(os.path.dirname(output_file)).mkdir(exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f'Metrics saved to {output_file}')
        
        return df
    
    def calculate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for network metrics.
        
        Args:
            df: DataFrame with network metrics
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        # Basic statistics by prompt type
        if 'prompt_type' in df.columns:
            summary['by_prompt_type'] = df.groupby('prompt_type')[
                ['nodes', 'edges', 'density', 'clustering', 'avg_deg']
            ].agg(['mean', 'std', 'min', 'max']).round(3)
        
        # Correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            summary['correlation_matrix'] = df[numeric_cols].corr().round(3)
        
        # Temperature-based statistics
        if 'temperature' in df.columns:
            summary['by_temperature'] = df.groupby('temperature')[
                ['nodes', 'edges', 'density', 'clustering', 'avg_deg']
            ].mean().round(3)
        
        return summary
    
    def compare_prompt_types(self, df: pd.DataFrame) -> Dict:
        """
        Compare metrics between different prompt types.
        
        Args:
            df: DataFrame with network metrics
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        if 'prompt_type' in df.columns:
            unique_types = df['prompt_type'].unique()
            
            if len(unique_types) >= 2:
                metrics = ['nodes', 'edges', 'density', 'avg_deg', 'clustering']
                
                for metric in metrics:
                    if metric in df.columns:
                        type_means = df.groupby('prompt_type')[metric].mean()
                        comparison[metric] = {
                            'means': type_means.to_dict(),
                            'difference': type_means.diff().dropna().to_dict()
                        }
        
        return comparison
    
    def analyze_single_network(self, file_path: str) -> Optional[Dict]:
        """
        Analyze a single network file.
        
        Args:
            file_path: Path to the network file
            
        Returns:
            Dictionary with network metrics or None if failed
        """
        return self.analyze_edge_list(file_path)
    
    def get_network_overview(self, df: pd.DataFrame) -> str:
        """
        Get a text overview of the network analysis results.
        
        Args:
            df: DataFrame with network metrics
            
        Returns:
            String with overview text
        """
        overview = []
        overview.append("=== NETWORK ANALYSIS OVERVIEW ===\n")
        
        # Basic statistics
        overview.append(f"Total networks analyzed: {len(df)}")
        
        if 'prompt_type' in df.columns:
            type_counts = df['prompt_type'].value_counts()
            overview.append(f"Networks by prompt type: {type_counts.to_dict()}")
        
        # Metric ranges
        numeric_cols = ['nodes', 'edges', 'density', 'clustering', 'avg_deg']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if available_cols:
            overview.append("\nMetric ranges:")
            for col in available_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                overview.append(f"  {col}: {min_val:.3f} - {max_val:.3f} (mean: {mean_val:.3f})")
        
        return "\n".join(overview)