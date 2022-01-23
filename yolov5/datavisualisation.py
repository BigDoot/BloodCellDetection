import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

# Load data
data = pd.read_csv('annotations.csv',engine='python')

images = data.image.unique()
d = data[data['image'].isin({'image-1.png'})]
print(d)

