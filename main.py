import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Load the Excel file
df = pd.read_excel('Maths.csv')

# Step 2: Drop any missing values (precautionary)
df = df.dropna()

# Step 3: Convert categorical columns into numerical using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 4: Separate features and target
X = df_encoded.drop('G3', axis=1)  # Features
y = df_encoded['G3']               # Target: Final grade

# Step 5: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict and evaluate the model
y_pred = model.predict(X_test)

# Step 8: Print evaluation metrics
print("📊 R² Score:", round(r2_score(y_test, y_pred), 3))
print("📉 Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))

import matplotlib.pyplot as plt

# Plot and save the prediction graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([0, 20], [0, 20], 'r--')  # Ideal line
plt.xlabel("Actual Final Scores (G3)")
plt.ylabel("Predicted Final Scores")
plt.title("Actual vs Predicted Final Grades")
plt.grid(True)
plt.tight_layout()

# Save in current folder
plt.savefig("plot.png")
print("📁 Plot saved as plot.png in current folder")
