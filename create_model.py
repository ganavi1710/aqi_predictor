from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# Sample training data
X = np.array([
    [50, 1, 20],
    [100, 2, 40],
    [150, 3, 60],
    [200, 4, 80]
])

y = np.array([50, 100, 150, 200])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
import os

file_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
pickle.dump(model, open(file_path, 'wb'))

print("Model created successfully!")
