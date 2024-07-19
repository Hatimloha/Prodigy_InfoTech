**Prodigy_task_2**

# Import necessary libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

```
# Import CSV & Read
```python
titanic_df = pd.read_csv("/content/train (1).csv")

```
# Display the first few rows of the dataset
```python
print(titanic_df.head())

```
# Output
```python
 PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S

```
# Check for missing values
```python
print(titanic_df.isnull().sum())

```
# Output
```python
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64

```
# Handling missing values
# For example, fill missing values in the 'Age' column with the median
```python
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

```
# Explore the distribution of numerical features
```python
sns.histplot(titanic_df['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.show()

# Output:  Graph will be created

```
# Explore the distribution of categorical features
```python
sns.countplot(x='Sex', data=titanic_df)
plt.title('Distribution of Gender')
plt.show()

# Output:  Graph will be created
```
# Explore the relationship between variables
```python
sns.scatterplot(x='Age', y='Fare', data=titanic_df, hue='Survived', palette='Set1')
plt.title('Scatterplot of Age and Fare by Survival')
plt.show()

Output: Visualization Graph will be created
```
# Explore survival rates across different categories
```python
sns.barplot(x='Pclass', y='Survived', data=titanic_df, hue='Sex', palette='viridis')
plt.title('Survival Rates by Passenger Class and Gender')
plt.show()

# Output:  Graph will be created
```
```python
numeric_df = titanic_df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
# Output: Correlation Matrix

