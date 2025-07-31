# Language Switching: Neural Representations and Cognitive Control

A computational neuroscience project exploring language switching mechanisms through multilingual transformer embeddings and simulated cognitive control signals.

## 🧠 Overview

This Jupyter notebook investigates how bilingual speakers switch between languages by analyzing the neural representation space of multilingual language models. We simulate language switching trajectories and model cognitive control signals that might govern bilingual language processing.

### Key Features

- **Multilingual Embedding Analysis**: Extract and visualize sentence representations from multilingual BERT and Sentence Transformers
- **Language Switching Simulation**: Model interpolation paths between English and Spanish sentence embeddings
- **Cognitive Control Modeling**: Simulate Anterior Cingulate Cortex (ACC)-like control signals during language switching
- **Interactive Visualizations**: Animated plots showing real-time language switching dynamics
- **Code-Switching Analysis**: Bidirectional language switching (EN → ES → EN) with control signal analysis

## 🚀 Getting Started

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge above to run the notebook directly in your browser with free GPU access.

### Option 2: Local Jupyter Setup

1. **Clone the repository:**
```bash
git clone https://github.com/George-Okello/language-switching.git
cd language-switching
```

2. **Create a virtual environment:**
```bash
python -m venv language_switching_env
source language_switching_env/bin/activate  # On Windows: language_switching_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install jupyter
pip install transformers datasets sentencepiece
pip install matplotlib seaborn scikit-learn
pip install sentence-transformers scipy
```

4. **Launch Jupyter:**
```bash
jupyter notebook Language_Switching.ipynb
```

### Option 3: Kaggle Notebook
Upload the `.ipynb` file to Kaggle for free GPU access and pre-installed libraries.

## 📋 Notebook Structure

The notebook is organized into the following sections:

### 1. **Setup and Dependencies** 
```python
# Install required packages
!pip install transformers datasets sentencepiece
!pip install matplotlib seaborn scikit-learn
```

### 2. **Data Loading and Preprocessing**
- Loads UN Parallel Corpus (English-Spanish)
- Cleans and filters sentence pairs
- Extracts aligned bilingual data

### 3. **Model Loading and Configuration**
- Multilingual BERT setup
- Sentence Transformer initialization  
- GPU optimization

### 4. **Embedding Extraction**
- CLS token embedding extraction
- Batch processing for efficiency
- Memory optimization techniques

### 5. **Visualization and Analysis**
- t-SNE dimensionality reduction
- Language clustering visualization
- Interpolation trajectory analysis

### 6. **Cognitive Control Simulation**
- ACC-like control signal modeling
- Code-switching pattern analysis
- Animated switching dynamics

### 7. **Results and Interpretation**
- Statistical analysis of switching patterns
- Cognitive load estimation
- Research implications

## 🔬 Scientific Background

This notebook is inspired by neuroscientific research on bilingual language processing:

- **Cognitive Control Theory**: How the brain manages competing languages
- **Neural Adaptation**: How multilingual representations adapt during switching  
- **ACC Function**: The role of anterior cingulate cortex in conflict monitoring

### Research Questions Explored

1. Do multilingual transformer models cluster languages in embedding space?
2. Can we model gradual transitions between languages?
3. What switching patterns require higher cognitive control?

## 📊 Expected Outputs

Running the notebook will generate:

### Visualizations
- **Language Clustering Plot**: t-SNE visualization of English vs Spanish embeddings
- **Switching Trajectories**: Interpolation paths between translation pairs
- **Control Signal Analysis**: Simulated ACC activation during switching
- **Animated Switching**: Real-time visualization of language transitions

### Data Analysis
- Cosine similarity matrices
- Control signal patterns
- Statistical summaries of switching dynamics

## 🎯 Key Functions and Code Cells

### Data Processing
```python
def clean_and_extract(dataset, lang1="en", lang2="es", max_samples=1000):
    # Filters and cleans parallel sentence pairs
```

### Embedding Extraction  
```python
def get_cls_embedding(text, tokenizer, model):
    # Extracts CLS token embeddings from multilingual BERT
```

### Visualization
```python
def animate_switching_paths(en_sentences, es_sentences):
    # Creates animated plots of language switching
```

### Control Modeling
```python
def simulate_control_signal(interpolated_vectors, en_emb, es_emb):
    # Models cognitive control demands during switching
```

## 💡 Usage Tips

### For Beginners
- Start by running all cells in order
- Adjust `sample_size` variables for faster execution during testing
- Use smaller datasets initially to understand the workflow

### For Researchers
- Modify language pairs by changing dataset parameters
- Experiment with different transformer models
- Adjust interpolation steps for finer analysis
- Save intermediate results for further analysis

### Performance Optimization
- Use GPU runtime in Colab for faster processing
- Reduce sample sizes for quicker iterations
- Cache embeddings to avoid recomputation

## 📈 Results Interpretation

### Language Clustering
The t-SNE plots should show distinct clustering of English and Spanish sentences, indicating that multilingual models maintain language-specific representations.

### Switching Trajectories  
Smooth interpolation paths suggest gradual transitions are possible in the embedding space, potentially modeling real cognitive switching processes.

### Control Signals
Higher control signal peaks during switching suggest increased cognitive demand, consistent with neuroscientific findings about bilingual processing.

## 🛠️ Customization Options

### Different Language Pairs
Replace dataset loading:
```python
dataset = load_dataset("un_pc", "en-fr")  # English-French
dataset = load_dataset("un_pc", "en-de")  # English-German
```

### Alternative Models
Try different transformers:
```python
model_name = "xlm-roberta-base"  # XLM-RoBERTa
model_name = "distilbert-base-multilingual-cased"  # DistilBERT
```

### Analysis Parameters
Adjust key parameters:
```python
sample_size = 500  # Number of sentence pairs
steps = 20  # Interpolation granularity  
perplexity = 30  # t-SNE parameter
```

## 🤝 Contributing

We welcome contributions! Ways to help:

- **Add new language pairs** from UN Parallel Corpus
- **Implement additional models** (XLM-R, mBERT variants)
- **Improve visualizations** with interactive plots
- **Add statistical tests** for significance testing
- **Optimize performance** for larger datasets

## 📚 Educational Use

This notebook is perfect for:

- **Computational Linguistics courses**: Understanding multilingual NLP
- **Cognitive Science classes**: Modeling bilingual processing
- **Machine Learning workshops**: Applied transformer analysis
- **Research projects**: Template for similar investigations

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce `sample_size` variables
- Use CPU instead of GPU for large datasets
- Process data in smaller batches

**Model Loading Failures:**
- Check internet connection for model downloads
- Verify transformers library version
- Use alternative model checkpoints

**Visualization Problems:**
- Install required plotting libraries
- Reduce data points for complex plots
- Check matplotlib backend settings

### Getting Help
1. Check the error messages in notebook output
2. Consult the [Transformers documentation](https://huggingface.co/docs/transformers/)
3. Open an issue with notebook output and error details

## 📖 Citation

If you use this notebook in your research:

```bibtex
@misc{language_switching_notebook_2024,
  title={Language Switching: Neural Representations and Cognitive Control},
  author={Your Name},
  howpublished={Jupyter Notebook, \url{https://github.com/yourusername/language-switching}},
  note={Originally developed in Google Colab},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify for research and educational purposes.

## 🙏 Acknowledgments

- **UN Parallel Corpus** for multilingual data
- **Hugging Face** for transformer models and datasets library
- **Google Colab** for providing free GPU access
- **Sentence Transformers** community for semantic similarity tools

---

**🔥 Pro Tip**: Start with the Colab version for the best experience with pre-configured environment and GPU access!
