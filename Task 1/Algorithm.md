**Prodigy_task_1**

```python
import matplotlib.pyplot as plt
import numpy as np

```
# Creating Dataset
ages = np.random.randint(18, 65, size=100)

```
# Create a histogram
plt.hist(ages, bins=10, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of contionus variable')

# Show the plot
plt.show()

> Histogram will create when we run the above script

```
import matplotlib.pyplot as plt

#sample data
genders = ['Male', 'Female', 'Non-Binary', 'Other']
counts = [120, 150, 20, 10]

```
# Create a bar chart
plt.bar(genders, counts, color=['blue', 'pink', 'purple', 'gray'])

# Add labels and title
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Bar chart of categorical varibale')

# Show the plot
plt.show()

> Bar chart will create when we run the above script
