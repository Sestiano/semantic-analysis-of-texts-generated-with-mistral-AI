# Semantic Analysis of Texts Generated with Mistral AI

A comprehensive research pipeline for analyzing the semantic structure of AI-generated texts using advanced network analysis, transformer-based embeddings, and robust statistical methods.


## Overview

This project investigates how generation parameters (temperature and prompt complexity) affect the semantic structure of texts produced by Mistral AI. The study employs state-of-the-art natural language processing techniques, including BERTScore for semantic similarity, bootstrap statistical inference, and network analysis to provide a rigorous examination of AI text generation patterns.

## Key Features

### Advanced Semantic Analysis
- **BERTScore Integration**: Contextual semantic similarity using Microsoft DeBERTa-XLarge-MNLI
- **Topic Modeling**: Latent Dirichlet Allocation for thematic analysis
- **Semantic Coherence**: Cross-prompt and temperature-based coherence analysis
- **Network Construction**: EmoAtlas-based semantic networks with enhanced metrics

### Robust Statistical Framework
- **Bootstrap Resampling**: 1000-sample bootstrap for confidence intervals
- **Effect Size Calculation**: Cohen's d and Hedges' g for practical significance
- **Multiple Comparison Correction**: False Discovery Rate control
- **Enhanced Sample Size**: 100 completions per condition for statistical power

### Comprehensive Network Analysis
- **Centrality Measures**: Degree, betweenness, closeness, eigenvector, PageRank
- **Community Detection**: Louvain and Leiden algorithms
- **Motif Analysis**: Pattern identification in semantic networks
- **Path Metrics**: Average path length and network diameter

## Architecture

The project follows a modular design with centralized configuration:

### Core Modules
- `mistral_generator.py`: Text generation with Mistral AI API
- `text_preprocessor.py`: SpaCy-based text cleaning and preprocessing
- `network_builder.py`: EmoAtlas semantic network construction
- `network_analyzer.py`: Comprehensive network metrics calculation
- `statistical_tests.py`: Advanced statistical analysis and hypothesis testing
- `visualizer.py`: High-quality visualization generation

### Main Analysis Notebook
- `Hakan4.ipynb`: Complete pipeline with centralized configuration

## Methodology

### Experimental Design
- **Prompt Types**: Complex narrative vs. simple vague prompts
- **Temperature Range**: 0.001 to 1.5 (7 levels)
- **Sample Size**: 100 text completions per condition
- **Statistical Power**: Bootstrap confidence intervals with 1000 samples

### Analysis Pipeline
1. **Text Generation**: Systematic generation with controlled parameters
2. **Preprocessing**: Tokenization, lemmatization, stop word removal
3. **Network Construction**: EmoAtlas-based semantic relationship mapping
4. **Semantic Analysis**: BERTScore similarity computation and topic modeling
5. **Statistical Testing**: Bootstrap inference and effect size calculation
6. **Visualization**: Multi-panel plots with publication-ready quality

### Statistical Rigor
- Bootstrap confidence intervals for all metrics
- Effect size estimation with practical interpretation
- Multiple comparison correction
- Permutation tests for null hypothesis validation
- Mixed-effects modeling for hierarchical data structure

## Requirements

### Core Dependencies
```
python>=3.8
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.1.0
networkx>=2.8
spacy>=3.4.0
```

### Advanced NLP Libraries
```
bert-score>=0.3.13
sentence-transformers>=2.2.0
gensim>=4.2.0
emoatlas>=1.0.0
mistralai>=0.1.0
```

### Visualization
```
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/Semantic-Analysis-of-Texts-Generated-with-Mistral-AI.git
cd Semantic-Analysis-of-Texts-Generated-with-Mistral-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

3. Configure Mistral API:
   - Obtain API key from Mistral AI
   - Set in notebook configuration section

## Usage

### Quick Start
1. Open `Hakan4.ipynb` in Jupyter Lab or VS Code
2. Configure parameters in Section 2 (Configuration and Setup)
3. Run cells sequentially for complete analysis

### Configuration Options
- Sample size: Adjust `n_completions` (default: 100)
- Bootstrap samples: Modify `n_bootstrap` (default: 1000)
- Semantic similarity threshold: Set `similarity_threshold` (default: 0.7)
- BERTScore model: Choose from available transformer models

### Output Structure
```
results/
├── network_metrics.csv          # Network analysis results
├── advanced_analysis_no_statsmodels.csv  # Statistical test results
└── pipeline_summary.txt         # Execution summary

bootstrap_results/
├── network_metrics_bootstrap.csv # Bootstrap confidence intervals
├── effect_sizes.csv            # Effect size calculations
└── semantic_coherence_bootstrap.csv # Semantic analysis bootstrap

semantic_analysis/
├── bertscore_similarities.csv   # BERTScore pairwise comparisons
├── topic_modeling_results.csv   # Topic analysis results
└── semantic_network_metrics.json # Semantic network properties

figures/
├── network_overview.png         # Network metrics visualization
├── correlation_heatmap.png      # Correlation analysis
└── comparative_analysis.png     # Cross-condition comparison
```

## Research Applications

### Academic Research
- Understanding creativity patterns in large language models
- Analyzing semantic coherence across generation parameters
- Comparative studies of AI text generation systems
- Development of automatic text quality assessment metrics

### Industry Applications
- Content generation quality control
- Prompt engineering optimization
- AI system evaluation and benchmarking
- Natural language generation parameter tuning

## Methodology Validation

This pipeline implements best practices for computational linguistics research:
- Adequate statistical power through large sample sizes
- Robust inference via bootstrap methods
- Effect size reporting for practical significance
- Reproducible analysis with centralized configuration
- Comprehensive documentation and visualization

## Citation

If you use this work in your research, please cite:

```bibtex
@software{semantic_analysis_mistral_2025,
  title={Semantic Analysis of Texts Generated with Mistral AI},
  author={[Author Names]},
  year={2025},
  url={https://github.com/username/Semantic-Analysis-of-Texts-Generated-with-Mistral-AI}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors and Acknowledgments

**Primary Author**: Sebastiano Franchini

This project originated from a hackathon collaboration with Roberto Passaro. The initial version can be found in the `backup_colab_first_version` folder. For Roberto's version of the project, visit his [GitHub profile](https://github.com/robertopassaro).

### Project History
- **Initial Version**: Hackathon collaboration (May, 2025)
- **Current Version**: Enhanced research pipeline with advanced statistical methods (July, 2025)

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate tests
4. Submit a pull request with detailed description
