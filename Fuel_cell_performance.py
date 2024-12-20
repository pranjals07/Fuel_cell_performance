# In[4]:


import pandas as pd

# Load the dataset
data = pd.read_csv("Fuel_cell_performance_data-Full.csv")


# In[5]:


# roll number ending in 4 (102203924)
target_col = "Target5"
data = data.drop(columns=["Target1", "Target2", "Target3", "Target4"])


# In[6]:


from sklearn.model_selection import train_test_split

X = data.drop(columns=[target_col])  # Features
y = data[target_col]                # Target variable

# Split into train and test sets (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[7]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Machine": SVR()
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    results[name] = mse

# Display results
print("Model Performance (MSE):", results)


# In[8]:


import matplotlib.pyplot as plt

plt.bar(results.keys(), results.values())
plt.title("Model Performance Comparison")
plt.ylabel("Mean Squared Error (MSE)")
plt.xticks(rotation=45)
plt.show()


# In[10]:


results_df = pd.DataFrame(list(results.items()), columns=["Model", "MSE"])
results_df.to_csv("model_results.csv", index=False)


# In[11]:


model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R² value:", r2)




