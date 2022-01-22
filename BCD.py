import pandas as pd
from sklearn.model_selection import train_test_split


# Load data
data = pd.read_csv('annotations.csv',engine='python')

# Display 5 random samples
print(data.sample(5))

# Split data into train and test sets in 8:2 ratio
images = data.image.unique()
train = images[:80]
test = images[80:]


