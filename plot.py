import pandas as pd
import matplotlib.pyplot as plt
import json

# Load the JSON data
file_path = "Richard_20250711_104551_blinks.json"
with open(file_path, "r") as f:
    blink_data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(blink_data)

# Calculate time in seconds from the first blink
start_time = df['timestamp_ms'].iloc[0]
df['time_seconds'] = (df['timestamp_ms'] - start_time) / 1000.0

# Raster-style blink plot
plt.figure(figsize=(14, 4))
plt.eventplot(df['time_seconds'], lineoffsets=1, colors='blue')
plt.title("Blink Events Over Time: Richard")
plt.xlabel("Time (seconds)")
plt.yticks([1], ["Blink"])
plt.grid(True)
plt.tight_layout()

plt.show()
