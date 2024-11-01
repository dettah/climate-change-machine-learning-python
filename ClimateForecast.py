import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data (rainfall and temperature)
rainfall = np.array([2442.5, 1599.4, 1879, 2132.6, 1905.6, 2251.4, 2115.0, 2588.6, 2032.8, 2246.7, 2256.8, 2233.1, 2668.7, 2143.4, 2144.6, 2004.0])
temperature = np.array([26.3, 24.4, 27.7, 27.6, 26.6, 27.0, 26.86, 26.8, 27.2, 27.1, 26.5, 26.4, 26.0, 26.17, 26.92, 27.5])

# Reshape the data for linear regression
rainfall = rainfall.reshape(-1, 1)

# Create and train a linear regression model
model = LinearRegression()
model.fit(rainfall, temperature)

# Make predictions using the model
predicted_temperature = model.predict(rainfall)

# Plot the data and the regression line
plt.scatter(rainfall, temperature, label="Actual Data")
plt.plot(rainfall, predicted_temperature, color='red', label="Linear Regression")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.title("Rainfall vs. Temperature")
plt.show()

# Predict temperature for a new rainfall value
new_rainfall = np.array([[70]])  
predicted_new_temperature = model.predict(new_rainfall)
print(f"Predicted temperature for {new_rainfall[0][0]} mm of rainfall: {predicted_new_temperature[0]:.2f} °C")
