**Prodigy_task_4**

# Import
```python
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
```

#Loading Dataset
```python
data = pd.read_csv('/content/twitter_training.csv')
```

```python
data.head()
```

# Output
```python
	2401	Borderlands	Positive	im getting on borderlands and i will murder you all ,
0	2401	Borderlands	Positive	I am coming to the borders and I will kill you...
1	2401	Borderlands	Positive	im getting on borderlands and i will kill you ...
2	2401	Borderlands	Positive	im coming on borderlands and i will murder you...
3	2401	Borderlands	Positive	im getting on borderlands 2 and i will murder ...
4	2401	Borderlands	Positive	im getting into borderlands and i can murder y...
```

# data.tail()
```python
2401	Borderlands	Positive	im getting on borderlands and i will murder you all ,
74676	9200	Nvidia	Positive	Just realized that the Windows partition of my...
74677	9200	Nvidia	Positive	Just realized that my Mac window partition is ...
74678	9200	Nvidia	Positive	Just realized the windows partition of my Mac ...
74679	9200	Nvidia	Positive	Just realized between the windows partition of...
74680	9200	Nvidia	Positive	Just like the windows partition of my Mac is l...
```

```python
col_names = ['ID', 'Entity', 'Sentiment', 'Content']
df = pd.read_csv('twitter_training.csv', names=col_names)
```

```python
df.head()
```

# Output
```python
ID	Entity	Sentiment	Content
0	2401	Borderlands	Positive	im getting on borderlands and i will murder yo...
1	2401	Borderlands	Positive	I am coming to the borders and I will kill you...
2	2401	Borderlands	Positive	im getting on borderlands and i will kill you ...
3	2401	Borderlands	Positive	im coming on borderlands and i will murder you...
4	2401	Borderlands	Positive	im getting on borderlands 2 and i will murder ...
```

```python
df.shape
```

```python
df.describe(include='all')
```

# Output
```python
ID	Entity	Sentiment	Content
count	74682.000000	74682	74682	73996
unique	NaN	32	4	69491
top	NaN	TomClancysRainbowSix	Negative	At the same time, despite the fact that there ...
freq	NaN	2400	22542	172
mean	6432.586165	NaN	NaN	NaN
std	3740.427870	NaN	NaN	NaN
min	1.000000	NaN	NaN	NaN
25%	3195.000000	NaN	NaN	NaN
50%	6422.000000	NaN	NaN	NaN
75%	9601.000000	NaN	NaN	NaN
max	13200.000000	NaN	NaN	NaN
```

# Data Cleaning
```python
df.isnull().sum()
```

# Output
```python
ID           0
Entity       0
Sentiment    0
Content      0
dtype: int64
```

```python
df.duplicated().sum()
```

```python
df.drop_duplicates(inplace=True)
df.duplicated().sum()
```

```python
df.shape
```

```python
sentiment_counts = df['Sentiment'].value_counts()
sentiment_counts
```

# Output
```python
Sentiment
Negative      21698
Positive      19713
Neutral       17708
Irrelevant    12537
Name: count, dtype: int64
```

```python
plt.figure(figsize=(6, 3))
sentiment_counts.plot(kind='bar', color=['red', 'green', 'yellow', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
plt.show()

# Output: Sentiment Distribution Graph
```

# Analysis On particular brand or company
```python
brand_data = df[df['Entity'].str.contains('Microsoft', case=False)]
brand_sentiment_counts = brand_data['Sentiment'].value_counts()
brand_sentiment_counts
```

# Output
```python
Sentiment
Neutral       816
Negative      748
Positive      573
Irrelevant    167
Name: count, dtype: int64
```

```python
plt.figure(figsize=(6, 6))
plt.pie(brand_sentiment_counts, labels=brand_sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution for Microsoft')
plt.show()
```

# Output: Sentiment Distribution for Microsoft Pie Chart
