import pandas as pd
from sdv.single_table import CTGANSynthesizer

print("Starting data synthesis process...")

# --- 1. Load and Preprocess the Original Data ---
# This section uses the same preprocessing steps from your model_trainer.py for consistency.
try:
    original_data = pd.read_csv('oasis_cross-sectional.csv')
    print("Original dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'oasis_cross-sectional.csv' not found. Please make sure it's in the same folder.")
    exit()

# Handle missing values using the median
original_data['SES'] = original_data['SES'].fillna(original_data['SES'].median())
original_data['MMSE'] = original_data['MMSE'].fillna(original_data['MMSE'].median())

# Encode the 'M/F' column
original_data['M/F'] = original_data['M/F'].replace({'M': 1, 'F': 0})

# Create the binary 'Dementia' target variable from CDR
original_data['Dementia'] = (original_data['CDR'] > 0).astype(int)

# Select only the columns needed for the model
model_columns = ['M/F', 'Age', 'Educ', 'SES', 'MMSE', 'Dementia']
preprocessed_data = original_data[model_columns]
print("Data preprocessing complete.")

# --- 2. Train the Synthesizer ---
# We use a CTGAN (Conditional Tabular GAN) model, which is excellent for tabular data.
synthesizer = CTGANSynthesizer(primary_key=None, epochs=500, verbose=True)

print("Training the synthesizer... This may take a few minutes.")
synthesizer.fit(preprocessed_data)
print("Synthesizer training complete.")

# --- 3. Generate Synthetic Data ---
print("Generating 10,000 new data samples...")
synthetic_data = synthesizer.sample(num_rows=10000)
print("Synthetic data generated.")

# --- 4. Save the New Dataset ---
output_filename = 'synthetic_dementia_dataset_10000.csv'
synthetic_data.to_csv(output_filename, index=False)

print("-" * 50)
print(f"✅ Success! A new dataset with 10,000 samples has been saved as '{output_filename}'")
print("-" * 50)

# Display a sample of the new data
print("\nSample of the newly generated data:")
print(synthetic_data.head())