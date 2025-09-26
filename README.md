# Waste Material Segregation using CNN

A deep learning project that implements an effective waste material segregation system using Convolutional Neural Networks (CNNs) to categorize waste into distinct groups, enhancing recycling efficiency and promoting sustainable waste management practices.

## Objective

The objective of this project is to implement an effective waste material segregation system using convolutional neural networks (CNNs) that categorises waste into distinct groups. This process enhances recycling efficiency, minimises environmental pollution, and promotes sustainable waste management practices.

### Key Goals:
- Accurately classify waste materials into categories like cardboard, glass, paper, and plastic
- Improve waste segregation efficiency to support recycling and reduce landfill waste
- Understand the properties of different waste materials to optimise sorting methods for sustainability

## Dataset

The dataset consists of images of common waste materials categorized into **7 classes**:

1. **Food Waste** - Coffee grounds, teabags, fruit peels, etc.
2. **Metal** - Aluminum cans, metal containers, etc.
3. **Paper** - Newspapers, magazines, documents, etc.
4. **Plastic** - Bottles, containers, packaging materials, etc.
5. **Other** - Miscellaneous waste items
6. **Cardboard** - Boxes, packaging materials, etc.
7. **Glass** - Bottles, jars, containers, etc.

### Dataset Structure:
- Multiple folders, each representing a specific waste class
- Images within each folder belong to that particular category
- Items are not further subcategorized within each class

## Getting Started

### Prerequisites

Make sure you have the following installed:
- Python 3.7+
- Jupyter Notebook or Google Colab
- Required libraries (see Installation section)

### Installation

Install the required dependencies:

```bash
pip install tensorflow
pip install keras
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install PIL
```

### Usage

1. Clone this repository:
```bash
git clone https://github.com/Bhuvanshree922/Waste_Segregation_CNN.git
cd Waste_Segregation_CNN
```

2. Open the Jupyter notebook:
```bash
jupyter notebook CNN_Waste_Segregation_Starter.ipynb
```

3. If using Google Colab, upload the notebook and run the cells sequentially.

## Model Architecture

The project implements two approaches:

### 1. Custom CNN Model
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for dimension reduction
- Dropout layers for regularization
- Dense layers for classification
- Softmax activation for multi-class classification

### 2. Transfer Learning Model
- Uses pre-trained models (like VGG16, ResNet, etc.)
- Fine-tuned for waste classification task
- Achieves better performance with less training time

## Model Performance

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Loss**: Training and validation loss curves
- **Confusion Matrix**: Detailed class-wise performance
- **Classification Report**: Precision, recall, and F1-score for each class

## Project Workflow

1. **Data Loading**: Load and organize the waste image dataset
2. **Data Preprocessing**: 
   - Image resizing and normalization
   - Data augmentation for better generalization
3. **Model Building**: 
   - Custom CNN architecture design
   - Transfer learning implementation
4. **Model Training**: 
   - Training with validation split
   - Monitoring accuracy and loss metrics
5. **Model Evaluation**: 
   - Performance analysis on test data
   - Visualization of results

## Features

- **Image Classification**: Automated waste material classification
- **Data Visualization**: Comprehensive plots for data analysis
- **Model Comparison**: Custom CNN vs Transfer Learning approaches
- **Performance Metrics**: Detailed evaluation with multiple metrics
- **Scalable Architecture**: Easy to extend for additional waste categories

## Environmental Impact

This project contributes to environmental sustainability by:
- Improving waste sorting accuracy
- Reducing manual sorting effort
- Enhancing recycling efficiency
- Minimizing contamination in recycling streams
- Supporting automated waste management systems

