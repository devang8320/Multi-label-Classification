# 📊 Multi-Label Classification with Voting Classifier

## 🚀 Project Overview
This project focuses on multi-label classification using a **Voting Classifier** approach, combining **Support Vector Machine (SVM)** and **Decision Tree Classifier**. The model is optimized using **GridSearchCV** and incorporates **class weighting** to address imbalanced data. Additionally, the dataset undergoes preprocessing, feature analysis, and visualization to improve model performance.

## 🎯 Features
### 🔢 Data Processing & Feature Engineering
- Merges multiple feature sets from different files.
- Handles missing values by replacing them with the mean.
- Normalizes and scales feature data.

### 🔍 Model Training & Optimization
- Implements **Voting Classifier** with SVM and Decision Tree.
- Uses **MultiOutputClassifier** for multi-label prediction.
- Applies **class weighting** to address data imbalance.
- Utilizes **GridSearchCV** for hyperparameter tuning.
- Saves the best model using `joblib`.

### 📊 Performance Evaluation
- Computes **F1 scores** for each class and overall average.
- Calculates **standard deviation** of F1 scores.
- Performs **cross-validation** to assess model stability.
- Exports predicted labels to a CSV file.

### 📈 Data Visualization & Analysis
- **Label Distribution**: Bar charts showing class frequencies.
- **Feature Distribution**: Histograms and box plots.
- **Pairwise Relationships**: Pairplot analysis for feature relationships.
- **Label Correlations**: Heatmaps to identify dependencies.

## 🏗️ Technology Stack
| Category          | Technologies Used |
|------------------|-----------------|
| Programming Language | Python |
| Machine Learning | scikit-learn |
| Data Handling | Pandas, NumPy |
| Model Storage | Joblib |
| Visualization | Matplotlib, Seaborn |

## 🚀 Installation & Setup
### 1️⃣ Clone the repository:
```bash
git clone https://github.com/yourusername/multi-label-classification.git
cd multi-label-classification
```
### 2️⃣ Install dependencies:

### 3️⃣ Prepare Data:
- Place dataset files (`PCA_R3_Normalized_ZScore.csv`, `labels_train.csv`, etc.) in the project directory.
- Ensure proper column structure in the dataset.

### 4️⃣ Run the Training Script:
```bash
python FinalClassifier.py
```
### 5️⃣ Make Predictions:
```bash
python FinalClassifier.py
```
### 6️⃣ Visualize Data:
```bash
python data_analyse.py
```

## 📬 Contact
👤 **Devang Vasani**  
📧 Email: devangvasani8320@gmail.com  

💡 Built with passion for machine learning and data-driven solutions! 🚀

