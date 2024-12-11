import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
labels_df = pd.read_csv('Group_02.csv')
features_df = pd.read_csv('R2_test.csv')
labels_df.head()
features_df.head()
# Calculate label frequencies
label_frequencies = labels_df.sum()
print(label_frequencies)

# Visualize label frequencies
plt.figure(figsize=(10, 6))
label_frequencies.plot(kind='bar')
plt.title('Label Distribution')
plt.xlabel('Class Index')
plt.ylabel('Number of Samples')
plt.show()
# Calculate the number of classes per sample
num_classes_per_sample = labels_df.sum(axis=1)

# # # Visualize the distribution
plt.figure(figsize=(19, 4))
sns.countplot(num_classes_per_sample)
plt.title('Distribution of Number of Classes per Sample')
plt.xlabel('Number of Classes')
plt.ylabel('Number of Samples')
plt.show()
# # Check for label imbalances
label_imbalances = label_frequencies / len(labels_df)
print('Label Imbalances:')
print(label_imbalances)
# Display basic descriptive statistics for each feature
features_df.describe()
# Visualize the distribution of a specific feature (e.g., the first feature)
plt.figure(figsize=(8, 5))
sns.histplot(features_df.iloc[:, 0], bins=30, kde=True)
plt.title('Distribution of Feature 1')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.show()
# # # Visualize outliers using box plots for a few features
plt.figure(figsize=(12, 8))
sns.boxplot(data=features_df.iloc[:, :5])
plt.title('Box Plots for Features 1-5')
plt.xlabel('Feature Index')
plt.ylabel('Feature Value')
plt.show()
# Visualize pairwise relationships between a subset of features
sns.pairplot(features_df.iloc[:, :4])
plt.suptitle('Pairwise Relationships between Features 1-4', y=1.02)
plt.show()
# Check for missing values in the feature set
missing_values = features_df.isnull().sum()
print('Missing Values:')
print(missing_values)
# # # In[7]:
# #
# #
# # # Visualize label correlations using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(features_df.corr(), cmap='coolwarm', annot=True)
plt.title('Label Correlation Heatmap')
plt.show()

# #
# # # In[8]:
# #
# #
# # # Display basic descriptive statistics for each feature
features_df.describe()
#
# # try 22/11/2023
#
import pandas as pd
#
# # Load feature sets
features_se4 = pd.read_csv('R4_train.csv')
features_se1 = pd.read_csv('R1_train.csv')
features_se2 = pd.read_csv('R2_train.csv')
features_se3 = pd.read_csv('R3_train.csv')
features_se5 = pd.read_csv('R5_train.csv')
features_se6 = pd.read_csv('R6_train.csv')

# # ... Repeat for feature_set3 to feature_set6
# #
# # Assuming 'sample_id' is a common identifier in all sets
# # Merge dataframes on the 'sample_id' column
combined_features = pd.merge(features_se1, features_se2, on='sample_id', how='inner')
combined_features = pd.merge(combined_features, features_se3, on='sample_id', how='inner')
combined_features = pd.merge(combined_features, features_se4, on='sample_id', how='inner')
combined_features = pd.merge(combined_features, features_se5, on='sample_id', how='inner')
combined_features = pd.merge(combined_features, features_se6, on='sample_id', how='inner')
#
feature_sets = [pd.read_csv(f'R{i}_train.csv') for i in range(1, 7)]
#
# # Concatenate along columns (axis=1)
combined_features = pd.concat(feature_sets, axis=1)

combined_features = combined_features.fillna(combined_features.mean())
#
# # ... Repeat for feature_set4 to feature_set6
#
# # The resulting dataframe 'combined_features' now contains all features from different sets for each sample
# # Assuming 'sample_id' is the common identifier in the label file
labels = pd.read_csv('labels_train.csv')
#
# # Merge combined features with labels
import pandas as pd
#
# # Load label file
#
#
# # Load combined feature file
#
#
# # Concatenate along columns
merged_data = pd.concat([combined_features, labels], axis=1)
# print(merged_data.head())
# print(merged_data.head())
from sklearn.model_selection import train_test_split

# # Assuming your data is stored in the 'merged_data' DataFrame
X = merged_data.drop(columns=['sample_id', 'label_column'])
y = merged_data['label_column']
# Assuming merged_data has 5000 features and 19 label columns

# Assuming merged_data has 5000 features and 19 label columns
X = merged_data.iloc[:, :-19].copy()
y = merged_data.iloc[:, -19:].copy()
print(X.head())
print(y.head())
#
#
#
# #
