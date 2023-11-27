import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

# Define the file path
file_path = r"C:\Users\Guy.Woumfouo\OneDrive - Beusa Energy, LLC\Desktop\Capstone2\Henry\025-1 Input Shaft Failure Henry Ratio Data.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)

# Convert the "Timestamp" column to datetime type
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set "Timestamp" column as the index
df.set_index('Timestamp', inplace=True)

# Group the data every 5 seconds using the column "Henry10-1"
grouped_data = df['Henry10-1'].resample('5S').mean().reset_index()

# Calculate the absolute differences between consecutive data points
time_diff = grouped_data['Timestamp'].diff()
value_diff = grouped_data['Henry10-1'].diff().abs()

# Calculate the time interval between data points in seconds
time_interval_seconds = time_diff.dt.total_seconds().mean()

# Compute the Fourier Transform of the signal
frequencies, power_spectrum = welch(value_diff, fs=1.0 / time_interval_seconds)

# Calculate the spectral entropy
spectral_entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))

# Create a distance-based plot
plt.figure(figsize=(10, 6))
plt.plot(grouped_data['Timestamp'], value_diff, label='Absolute Value Difference')
plt.xlabel('Timestamp')
plt.ylabel('Absolute Value Difference')
plt.title(f'Distance-Based Plot for Henry10-1\nSpectral Entropy: {spectral_entropy:.2f}')
plt.grid(True)

plt.legend()
plt.show()
