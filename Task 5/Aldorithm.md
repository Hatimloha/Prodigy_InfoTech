**Prodigy_task_5**

# Importing necessary libraries
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
```

# Loading the data
```python
data = pd.read_csv('/content/Road_Accidents_2017-Annuxure_Tables_3.csv')
data = pd.read_csv('/content/Road_Accidents_2017-Annuxure_Tables_4.csv')
```

```python
data.head()
```

# Displaying the first few rows of each dataframe
```python
print("Data1 Head:")
print(data1.head())
print("\nData2 Head:")
print(data2.head())
```

# Checking for missing values 
```python
print("\nMissing values in Data1:")
print(data1.isnull().sum())
print("\nMissing values in Data2:")
print(data2.isnull().sum())
```

# Checking the structure of the datasets
```python
print("\nData1 Info:")
print(data1.info())
print("\nData2 Info:")
print(data2.info())
```

# Merging the datasets (assuming a common column named 'id')
```python
merged_data = pd.merge(data1, data2, on='id', how='inner')
```

# Handling missing values (drop rows with missing values for simplicity) 
```python
merged_data = merged_data.dropna()
```

# Encoding categorical variables
```python
label_encoder = LabelEncoder()
for column in merged_data.select_dtypes(include=['object']).columns:
    merged_data[column] = label_encoder.fit_transform(merged_data[column])
```

# Standardizing the data
```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data)
```

# Converting the scaled data back to a DataFrame
```python
scaled_data = pd.DataFrame(scaled_data, columns=merged_data.columns)
```

```python
print("\nProcessed Data Head:")
print(scaled_data.head())
```

