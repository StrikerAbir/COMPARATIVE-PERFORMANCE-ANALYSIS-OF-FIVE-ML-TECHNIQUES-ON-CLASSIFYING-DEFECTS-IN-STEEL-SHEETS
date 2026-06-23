# Comparative Performance Analysis of Five ML Techniques on Classifying Defects in Steel Sheets from Images

## 📋 Project Overview

This repository contains a comprehensive comparative analysis of five machine learning techniques for automated defect classification in steel sheet images. The project evaluates multiple deep learning and machine learning approaches to determine their effectiveness, accuracy, and efficiency in identifying various defects in industrial steel production.

## 🎯 Objective

To benchmark and compare the performance of five distinct machine learning techniques in classifying defects in steel sheet images, identifying the most suitable approach for production-level implementation.

## 📊 ML Techniques Evaluated

This project implements and compares the following five techniques:

1. **Convolutional Neural Networks (CNN)** - Deep learning baseline for image classification
2. **Transfer Learning (Pre-trained Models)** - Leveraging pre-trained architectures (e.g., VGG, ResNet)
3. **Support Vector Machine (SVM)** - Classical machine learning approach with feature extraction
4. **Random Forest** - Ensemble learning method with extracted image features
5. **Gradient Boosting** - Advanced ensemble technique for classification

## 📁 Repository Structure

```
.
├── README.md                          # Project documentation
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset analysis and visualization
│   ├── 02_data_preprocessing.ipynb    # Data cleaning and augmentation
│   ├── 03_cnn_model.ipynb             # CNN implementation
│   ├── 04_transfer_learning.ipynb     # Transfer learning approach
│   ├── 05_svm_classification.ipynb    # SVM implementation
│   ├── 06_random_forest.ipynb         # Random Forest classifier
│   ├── 07_gradient_boosting.ipynb     # Gradient Boosting model
│   └── 08_performance_comparison.ipynb # Comprehensive results analysis
├── data/
│   ├── raw/                           # Original dataset
│   ├── processed/                     # Preprocessed data
│   └── README.md                      # Data documentation
├── models/
│   ├── cnn_model.h5
│   ├── transfer_learning_model.h5
│   └── other_models.pkl
├── results/
│   ├── performance_metrics.csv
│   ├── comparison_charts.png
│   └── detailed_analysis.txt
└── requirements.txt                   # Python dependencies
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- GPU support (recommended for faster training)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/StrikerAbir/COMPARATIVE-PERFORMANCE-ANALYSIS-OF-FIVE-ML-TECHNIQUES-ON-CLASSIFYING-DEFECTS-IN-STEEL-SHEETS.git
cd COMPARATIVE-PERFORMANCE-ANALYSIS-OF-FIVE-ML-TECHNIQUES-ON-CLASSIFYING-DEFECTS-IN-STEEL-SHEETS
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 📦 Dependencies

Key libraries used in this project:
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Classical ML algorithms and metrics
- **OpenCV** - Image processing
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **XGBoost** - Gradient boosting implementation

See `requirements.txt` for complete list.

## 📈 Key Findings & Metrics

### Evaluation Metrics
- **Accuracy** - Overall classification correctness
- **Precision** - True positive rate among predicted positives
- **Recall** - True positive rate among actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Detailed classification breakdown
- **Training Time** - Model training duration
- **Inference Time** - Prediction speed

### Expected Results Summary
| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| CNN | - | - | - | - | - |
| Transfer Learning | - | - | - | - | - |
| SVM | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| Gradient Boosting | - | - | - | - | - |

*Results to be populated after model evaluation*

## 🚀 Usage

### Running Individual Models

1. **CNN Model**
```bash
jupyter notebook notebooks/03_cnn_model.ipynb
```

2. **Transfer Learning**
```bash
jupyter notebook notebooks/04_transfer_learning.ipynb
```

3. **Traditional ML Models**
```bash
jupyter notebook notebooks/05_svm_classification.ipynb
jupyter notebook notebooks/06_random_forest.ipynb
jupyter notebook notebooks/07_gradient_boosting.ipynb
```

4. **Compare Results**
```bash
jupyter notebook notebooks/08_performance_comparison.ipynb
```

## 💡 Key Features

✅ **Comprehensive Comparison** - Side-by-side evaluation of all five techniques  
✅ **Production-Ready Code** - Well-documented, modular Jupyter notebooks  
✅ **Multiple Metrics** - Accuracy, precision, recall, F1-score, and more  
✅ **Visualization** - Detailed charts, confusion matrices, and performance plots  
✅ **Reproducibility** - Fixed random seeds for consistent results  
✅ **Data Augmentation** - Enhanced training datasets for robustness  

## 📊 Dataset Information

- **Source**: [Dataset information to be added]
- **Size**: [Number of images]
- **Classes**: [Number of defect types]
- **Split**: Training/Validation/Test ratio [to be specified]
- **Format**: Image format (JPG/PNG)

## 🔍 Defect Categories

The project focuses on classifying the following steel sheet defects:
- [Defect Type 1]
- [Defect Type 2]
- [Defect Type 3]
- [Defect Type 4]
- [Defect Type 5]

*(Update with actual defect types from your dataset)*

## 📝 Methodology

### Data Preprocessing
- Image resizing and normalization
- Data augmentation (rotation, flipping, brightness adjustment)
- Train-validation-test split

### Feature Engineering
- For traditional ML: Hand-crafted features or deep features from CNNs
- For deep learning: Automatic feature extraction through layers

### Model Training
- Cross-validation for robust evaluation
- Hyperparameter tuning using grid search
- Early stopping to prevent overfitting

## 🎓 Learning Outcomes

This project demonstrates:
- Implementation of diverse ML/DL techniques
- Computer vision application in industrial quality control
- Model comparison and selection best practices
- Performance optimization strategies

## 👨‍💼 Author

**StrikerAbir**  
GitHub: [@StrikerAbir](https://github.com/StrikerAbir)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset providers and contributors
- Open-source ML/DL community
- TensorFlow, Scikit-learn, and PyTorch communities

## 📧 Contact & Contributions

For questions, suggestions, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Contact the project maintainer

## 🔗 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide)
- [Steel Defect Detection Research Papers]

---

**Last Updated**: June 2026  
**Status**: Active Development
