import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

class DatasetLoader:
    def __init__(self, file_path):
        """Initialize with the path to the dataset."""
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}

    def load_data(self):
        """Load the CSV dataset and display sample data."""
        try:
            self.data = pd.read_csv(self.file_path)
            print("Available columns:", self.data.columns.tolist())
            print("\nSample data before preprocessing:")
            print(self.data.head())
            print(f"\nDataset loaded successfully with {self.data.shape[0]} records.")
        except FileNotFoundError:
            print("Error: Dataset file not found.")
            raise

    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                if self.data[column].dtype in ['int64', 'float64']:
                    # Impute numerical columns with mean
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                else:
                    # Impute categorical columns with mode
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)
        print("Missing values handled.")

    def encode_categorical(self, categorical_columns):
        """Encode categorical variables."""
        # Verify that all specified columns exist
        missing_cols = [col for col in categorical_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataset: {missing_cols}")
        
        for column in categorical_columns:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
            self.label_encoders[column] = le
        print("Categorical variables encoded.")

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """Split data into training and test sets."""
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Data split: {len(self.X_train)} training, {len(self.X_test)} test samples.")

    def save_preprocessed_data(self, output_path):
        """Save preprocessed dataset and display sample data."""
        self.data.to_csv(output_path, index=False)
        print("Sample data after preprocessing:")
        print(self.data.head())
        print(f"Preprocessed data saved to {output_path}.")

# Example usage
if __name__ == "__main__":
    # Initialize the loader
    loader = DatasetLoader("data.csv")
    
    # Load and preprocess the dataset
    loader.load_data()
    
    # Handle missing values
    loader.handle_missing_values()
    
    # Define categorical columns based on dataset
    categorical_cols = [
        "Gender", "Work Type", "Residence Type", "Smoking Status", 
        "Physical Activity", "Dietary Habits", "Alcohol Consumption", 
        "Education Level", "Income Level", "Region"
    ]
    loader.encode_categorical(categorical_cols)
    
    # Split data with 'Stroke Occurrence' as the target
    loader.split_data(target_column="Stroke Occurrence")
    
    # Save preprocessed data
    loader.save_preprocessed_data("preprocessed_data.csv")