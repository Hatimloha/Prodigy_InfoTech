**Prodigy_task_1**

```Import_library
import matplotlib.pyplot as plt
import numpy as np

```
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
