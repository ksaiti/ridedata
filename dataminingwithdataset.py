import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

# Load the dataset 
#rides = pd.read_csv('Rides_Data.csv') 
#drivers = pd.read_csv('Drivers_Data.csv')

# Explore the rides dataset 
#print(rides.head()) 
#print(rides.info()) 

# Explore the drivers dataset 
#print(drivers.head()) 
#print(drivers.info()) 

# Merge datasets on the common key (e.g., 'driver_id') 
#merged_data = pd.merge(rides, drivers, on='Driver_ID') 

# Visualize the distribution of ride ratings 
#plt.hist(merged_data['Rating'], bins=20) 
#plt.xlabel('Ride Rating') 
#plt.ylabel('Frequency') 
#plt.title('Distribution of Ride Ratings') 
#plt.show()

# Handle missing values
#merged_data = merged_data.dropna()

#print(merged_data.columns)
# Encode categorical variables (if any)
# Example: merged_data = pd.get_dummies(merged_data, columns=['categorical_column'])

# Define features and target variable
#X = merged_data[['Age', 'Distance_km', 'Duration_min']]  # Replace with relevant features
#y = merged_data['Average_Rating']

# Split the data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
#model = LinearRegression()
#model.fit(X_train, y_train)

# Predict on the test set
#y_pred = model.predict(X_test)

# Calculate performance metrics
#mse = mean_squared_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)

#print('Mean Squared Error:', mse)
#print('R^2 Score:', r2)


#import matplotlib.pyplot as plt

# Scatter plot for actual vs predicted ratings
#plt.figure(figsize=(10, 6))
#plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted Ratings')
#plt.scatter(y_test, y_test, alpha=0.5, color='green', label='Actual Ratings')
#plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Prediction Line')  # Line for perfect prediction
#plt.xlabel('Actual Ratings')
#plt.ylabel('Predicted Ratings')
#plt.title('Actual Ratings vs Predicted Ratings')
#plt.legend()
#plt.show()


# Calculate residuals
#residuals = y_test - y_pred

# Plot residuals
#plt.figure(figsize=(10, 6))
#plt.scatter(y_pred, residuals, alpha=0.5)
#plt.axhline(y=0, color='red', linestyle='--')
#plt.xlabel('Predicted Ratings')
#plt.ylabel('Residuals')
#plt.title('Residuals vs Predicted Ratings')
#plt.show()

# Scatter plot for actual vs predicted ratings with line for perfect fit
#plt.figure(figsize=(10, 6))
#plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted Ratings')
#plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Prediction Line')
#plt.xlabel('Actual Ratings')
#plt.ylabel('Predicted Ratings')
#plt.title('Actual Ratings vs Predicted Ratings')
#plt.legend()
#plt.show()
