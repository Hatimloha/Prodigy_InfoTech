**Prodigy_task_3**

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

# Loading data
```python
bank = pd.read_csv("/content/bank1.csv",sep=';')
```

```python
bank.head()
bank.shape
```

```python
bank.info()
```

# Output
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 41188 entries, 0 to 41187
Data columns (total 21 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   age             41188 non-null  int64  
 1   job             41188 non-null  object 
 2   marital         41188 non-null  object 
 3   education       41188 non-null  object 
 4   default         41188 non-null  object 
 5   housing         41188 non-null  object 
 6   loan            41188 non-null  object 
 7   contact         41188 non-null  object 
 8   month           41188 non-null  object 
 9   day_of_week     41188 non-null  object 
 10  duration        41188 non-null  int64  
 11  campaign        41188 non-null  int64  
 12  pdays           41188 non-null  int64  
 13  previous        41188 non-null  int64  
 14  poutcome        41188 non-null  object 
 15  emp.var.rate    41188 non-null  float64
 16  cons.price.idx  41188 non-null  float64
 17  cons.conf.idx   41188 non-null  float64
 18  euribor3m       41188 non-null  float64
 19  nr.employed     41188 non-null  float64
 20  y               41188 non-null  object 
dtypes: float64(5), int64(5), object(11)
memory usage: 6.6+ MB
```
# Statistical Analysis
```python
bank.describe()
```

# Output
```python
	age	duration	campaign	pdays	previous	emp.var.rate	cons.price.idx	cons.conf.idx	euribor3m	nr.employed
count	41188.00000	41188.000000	41188.000000	41188.000000	41188.000000	41188.000000	41188.000000	41188.000000	41188.000000	41188.000000
mean	40.02406	258.285010	2.567593	962.475454	0.172963	0.081886	93.575664	-40.502600	3.621291	5167.035911
std	10.42125	259.279249	2.770014	186.910907	0.494901	1.570960	0.578840	4.628198	1.734447	72.251528
min	17.00000	0.000000	1.000000	0.000000	0.000000	-3.400000	92.201000	-50.800000	0.634000	4963.600000
25%	32.00000	102.000000	1.000000	999.000000	0.000000	-1.800000	93.075000	-42.700000	1.344000	5099.100000
50%	38.00000	180.000000	2.000000	999.000000	0.000000	1.100000	93.749000	-41.800000	4.857000	5191.000000
75%	47.00000	319.000000	3.000000	999.000000	0.000000	1.400000	93.994000	-36.400000	4.961000	5228.100000
max	98.00000	4918.000000	56.000000	999.000000	7.000000	1.400000	94.767000	-26.900000	5.045000	5228.100000
```

# Checking Missing values
```python
bank.isnull().sum()
```

#
```python
age               0
job               0
marital           0
education         0
default           0
housing           0
loan              0
contact           0
month             0
day_of_week       0
duration          0
campaign          0
pdays             0
previous          0
poutcome          0
emp.var.rate      0
cons.price.idx    0
cons.conf.idx     0
euribor3m         0
nr.employed       0
y                 0
dtype: int64
```

# Checking for duplicates
```python
bank.duplicated().sum()
```

# Investigating these 12 duplicates
```python
bank[bank.duplicated()]
```

# Output
```python
	age	job	marital	education	default	housing	loan	contact	month	day_of_week	...	campaign	pdays	previous	poutcome	emp.var.rate	cons.price.idx	cons.conf.idx	euribor3m	nr.employed	y
1266	39	blue-collar	married	basic.6y	no	no	no	telephone	may	thu	...	1	999	0	nonexistent	1.1	93.994	-36.4	4.855	5191.0	no
12261	36	retired	married	unknown	no	no	no	telephone	jul	thu	...	1	999	0	nonexistent	1.4	93.918	-42.7	4.966	5228.1	no
14234	27	technician	single	professional.course	no	no	no	cellular	jul	mon	...	2	999	0	nonexistent	1.4	93.918	-42.7	4.962	5228.1	no
16956	47	technician	divorced	high.school	no	yes	no	cellular	jul	thu	...	3	999	0	nonexistent	1.4	93.918	-42.7	4.962	5228.1	no
18465	32	technician	single	professional.course	no	yes	no	cellular	jul	thu	...	1	999	0	nonexistent	1.4	93.918	-42.7	4.968	5228.1	no
20216	55	services	married	high.school	unknown	no	no	cellular	aug	mon	...	1	999	0	nonexistent	1.4	93.444	-36.1	4.965	5228.1	no
20534	41	technician	married	professional.course	no	yes	no	cellular	aug	tue	...	1	999	0	nonexistent	1.4	93.444	-36.1	4.966	5228.1	no
25217	39	admin.	married	university.degree	no	no	no	cellular	nov	tue	...	2	999	0	nonexistent	-0.1	93.200	-42.0	4.153	5195.8	no
28477	24	services	single	high.school	no	yes	no	cellular	apr	tue	...	1	999	0	nonexistent	-1.8	93.075	-47.1	1.423	5099.1	no
32516	35	admin.	married	university.degree	no	yes	no	cellular	may	fri	...	4	999	0	nonexistent	-1.8	92.893	-46.2	1.313	5099.1	no
36951	45	admin.	married	university.degree	no	no	no	cellular	jul	thu	...	1	999	0	nonexistent	-2.9	92.469	-33.6	1.072	5076.2	yes
38281	71	retired	single	university.degree	no	no	no	telephone	oct	tue	...	1	999	0	nonexistent	-3.4	92.431	-26.9	0.742	5017.5	no
12 rows × 21 columns
```
# Exploratory Data Analysis Age Distribution
```python

```
