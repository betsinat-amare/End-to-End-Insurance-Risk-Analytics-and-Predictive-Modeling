# src/preprocess.py
import pandas as pd

# Load raw data
data = pd.read_csv("../data/raw/MachineLearningRating_v3.txt", sep="\t")  # adjust separator if needed

# Display data info
print("Data Info:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Save processed data
data.to_csv("../data/processed/cleaned_data.csv", index=False)
print("\nProcessed data saved to data/processed/cleaned_data.csv")
