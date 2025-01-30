import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import os

# Load the dataset
def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Preprocess the dataset
def preprocess_dataset(dataset):
    # Drop unnecessary columns
    dataset = dataset.drop(columns=['Person ID'])

    # Encode categorical features
    categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure']
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
        label_encoders[column] = le

    # Encode target variable
    target_encoder = LabelEncoder()
    dataset['Sleep Disorder'] = target_encoder.fit_transform(dataset['Sleep Disorder'])

    # Normalize numerical features
    numerical_columns = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                         'Stress Level', 'Heart Rate', 'Daily Steps']
    scaler = StandardScaler()
    dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

    return dataset, label_encoders, target_encoder, scaler

# Split dataset into train, test, and non-IID client subsets
def split_dataset_non_iid(dataset, test_size=0.2, num_clients=5):
    # Separate features and target
    X = dataset.drop(columns=['Sleep Disorder'])
    y = dataset['Sleep Disorder']

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Shuffle training data
    train_data = X_train.copy()
    train_data['Sleep Disorder'] = y_train
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split training data into non-overlapping subsets for clients
    client_subsets = []
    client_data_size = len(train_data) // num_clients
    for i in range(num_clients):
        start_idx = i * client_data_size
        end_idx = start_idx + client_data_size if i < num_clients - 1 else len(train_data)
        client_data = train_data.iloc[start_idx:end_idx]
        client_X = client_data.drop(columns=['Sleep Disorder'])
        client_y = client_data['Sleep Disorder']
        client_subsets.append((client_X, client_y))

    return X_train, X_test, y_train, y_test, client_subsets

# Save client data to files for simulation
def save_client_data(client_subsets, output_dir="client_data"):
    os.makedirs(output_dir, exist_ok=True)
    for i, (X_client, y_client) in enumerate(client_subsets):
        client_dir = os.path.join(output_dir, f"client_{i+1}")
        os.makedirs(client_dir, exist_ok=True)
        X_client.to_csv(os.path.join(client_dir, "X_client.csv"), index=False)
        y_client.to_csv(os.path.join(client_dir, "y_client.csv"), index=False)

# Main function
def main():
    # File path to the dataset
    file_path = "Sleep_health_and_lifestyle_dataset.csv"

    # Load and preprocess the dataset
    dataset = load_dataset(file_path)
    dataset, label_encoders, target_encoder, scaler = preprocess_dataset(dataset)

    # Split the dataset
    X_train, X_test, y_train, y_test, client_subsets = split_dataset_non_iid(dataset)

    # Save client data
    save_client_data(client_subsets)

    # Save test data
    os.makedirs("test_data", exist_ok=True)
    X_test.to_csv("test_data/X_test.csv", index=False)
    y_test.to_csv("test_data/y_test.csv", index=False)

    print("Preprocessing complete. Non-IID client and test data saved.")

if __name__ == "__main__":
    main()
