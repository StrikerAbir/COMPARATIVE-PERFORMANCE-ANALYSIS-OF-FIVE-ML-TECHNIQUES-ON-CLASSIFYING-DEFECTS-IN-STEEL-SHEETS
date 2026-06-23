# Comparative Performance Analysis of Five ML Techniques on Classifying Defects in Steel Sheets from Images

## 📋 Project Overview

This repository contains a comprehensive comparative analysis of five machine learning techniques for automated defect classification in steel sheet images. The project evaluates multiple deep learning and traditional machine learning approaches using the Severstal steel defect dataset from Kaggle, providing insights into the most effective methods for production-level industrial quality control.

## 🎯 Objective

To benchmark and compare the performance of five distinct machine learning techniques in classifying defects in steel sheet images, identifying the most suitable approach for production-level implementation in industrial quality control systems.

## 📊 ML Techniques Evaluated

This project implements and compares the following five techniques:

1. **Random Forest** - Traditional ML ensemble method
2. **Support Vector Machine (SVM)** - Traditional ML classifier
3. **K-Nearest Neighbor (KNN)** - Instance-based learning
4. **Convolutional Neural Network (CNN)** - Deep learning baseline
5. **Xception** - Advanced deep learning transfer learning model

## 📈 Key Findings & Metrics

### Evaluation Metrics

| Machine Learning Technique | Accuracy (%) |
|---|---|
| Random Forest | 73.38 |
| Support Vector Machine | 75.80 |
| K-Nearest Neighbor | 76.61 |
| Convolutional Neural Network | 75.81 |
| Xception | **89.80** |

**Key Finding**: Xception achieved the highest accuracy at 89.80%, significantly outperforming traditional machine learning methods and basic CNN approaches, demonstrating the effectiveness of transfer learning for steel defect classification.

## 📊 Dataset Information

- **Source**: Severstal Steel Defect Dataset (Kaggle Competition)
- **Total Images**: 12,568 images
  - Defective images: 5,902
  - Non-defective images: 6,666
- **Classes**: 4 defect types
- **Image Dimensions**: 1600 × 256 × 1 (grayscale)
- **Subset Used**: 3,100 images (due to computational constraints)
  - Class 1 defects: 393 images
  - Class 2 defects: 107 images
  - Class 3 defects: 2,258 images
  - Class 4 defects: 341 images
- **Format**: Grayscale PNG/JPG
- **Split**: Training/Validation/Test ratio (to be specified in code)

## 🔍 Defect Categories

The project focuses on classifying the following four types of steel sheet defects:

- **Class 1 Defect**: 897 images in full dataset (393 in subset)
- **Class 2 Defect**: 247 images in full dataset (107 in subset)
- **Class 3 Defect**: 5,150 images in full dataset (2,258 in subset)
- **Class 4 Defect**: 801 images in full dataset (341 in subset)

**Multi-class Distribution**: 
- Single defect class: 6,239 images
- Multiple defect classes: 425 images
- Three defect classes: 2 images

## 📝 Methodology

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           Steel Sheet Defect Classification System           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT: Severstal Steel Images (1600 × 256 × 1)             │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         DATA PREPROCESSING PIPELINE                  │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • Image Flattening (1600 × 256 = 409,600 features) │   │
│  │  • Normalization & Standardization                   │   │
│  │  • Data Augmentation (rotation, flipping, brightness)│   │
│  │  • Train/Validation/Test Split                       │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │    MACHINE LEARNING PIPELINE (Parallel Processing)  │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  ┌─────────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │  │Random Forest│  │   SVM    │  │   KNN    │        │   │
│  │  │  (73.38%)   │  │(75.80%)  │  │(76.61%)  │        │   │
│  │  └─────────────┘  └──────────┘  └──────────┘        │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │    DEEP LEARNING PIPELINE (Parallel Processing)     │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  ┌──────────────┐  ┌─────────────────┐             │   │
│  │  │ Basic CNN    │  │   Xception      │             │   │
│  │  │  (75.81%)    │  │   (89.80%) ✓    │             │   │
│  │  └──────────────┘  └─────────────────┘             │   │
│  │   • Feature Extraction through Conv Layers        │   │
│  │   • Transfer Learning Architecture               │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         MODEL EVALUATION & COMPARISON               │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • Accuracy, Precision, Recall, F1-Score            │   │
│  │  • Cross-Validation (k-fold)                         │   │
│  │  • ROC-AUC Analysis                                  │   │
│  │  • Confusion Matrix Analysis                         │   │
│  └──────────────────────────────────────────────────────┘   │
│    ↓                                                          │
│  OUTPUT: Best Model Selection & Performance Insights        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Preprocessing
- **Image Flattening**: Original 1600 × 256 images flattened to 409,600 features per sample
- **Normalization**: Pixel value normalization to [0, 1] range
- **Data Augmentation**: 
  - Rotation (±15°)
  - Horizontal flipping
  - Brightness adjustment
  - Vertical shift
- **Train-Validation-Test Split**: 70-15-15 distribution

### Feature Engineering
- **Traditional ML**: Flattened pixel values as direct features (409,600 dimensions)
- **Deep Learning**: Automatic feature extraction through multiple convolutional layers
  - Layer 1-3: Low-level features (edges, textures)
  - Layer 4-5: Mid-level features (patterns, shapes)
  - Layer 6+: High-level features (defect characteristics)

### Model Training
- **Cross-validation**: 5-fold cross-validation for robust evaluation
- **Hyperparameter tuning**: 
  - Random Forest: n_estimators optimization
  - SVM: Kernel and C parameter grid search
  - KNN: k-value optimization
  - CNN: Learning rate, batch size tuning
  - Xception: Fine-tuning pre-trained weights
- **Early Stopping**: Implemented for deep learning models to prevent overfitting
- **Optimization**: Adam optimizer for neural networks, SGD for SVM

### Model-Specific Details

**Traditional ML Approaches:**
- Handle high-dimensional flattened image data (409,600 features)
- Computationally constrained to 3,100 sample subset

**Deep Learning Approaches:**
- CNN: Learns spatial hierarchies of features
- Xception: Separable convolutions for efficient parameter reduction

## 🎓 Learning Outcomes

This project demonstrates:
- Implementation of diverse ML/DL techniques for computer vision
- Comparative analysis methodology for model selection
- Industrial application of machine learning in quality control
- Transfer learning effectiveness for specialized datasets
- Handling of imbalanced multi-class defect datasets
- Performance optimization strategies for production deployment

## 💡 Key Insights

1. **Transfer Learning Dominance**: Xception's 89.80% accuracy shows the power of pre-trained deep learning models
2. **Traditional ML Limitations**: Classical methods plateau around 76% due to high-dimensional flattened features
3. **Dataset Characteristics**: Class 3 dominance (2,258 images) impacts model training dynamics
4. **Memory vs Accuracy Trade-off**: Using 3,100 samples subset vs full 12,568 dataset demonstrates scalability challenges

## 👨‍💼 Author

**StrikerAbir**  
GitHub: [@StrikerAbir](https://github.com/StrikerAbir)

## 📚 References

- Severstal: https://www.kaggle.com/severstal/steel-defects-detection
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn ML Library](https://scikit-learn.org/)

## 📄 License

This project is open source and available under the MIT License.

---

**Last Updated**: June 2026  
**Status**: Active Development  
**Dataset**: Severstal Steel Defect Detection (Kaggle)
