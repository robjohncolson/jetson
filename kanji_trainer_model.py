# kanji_trainer_model.py
# Trains a simple neural network on the collected Kanji stroke data.
# FIX: Handle multi-stroke data structure from kanji_dataset.json

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import traceback # For debugging

# --- Configuration ---
DATASET_FILENAME = "kanji_dataset.json"
MODEL_FILENAME = "kanji_classifier.pth"
MAX_TOTAL_POINTS = 600  # Fixed total number of points (x,y,p) per trial (adjust as needed)
# Example: If average trial has 3 strokes of 200 points, 600 is reasonable.
# Increase if Kanji are very complex or decrease if simpler.

# --- Load dataset ---
print(f"Loading dataset from {DATASET_FILENAME}...")
try:
    with open(DATASET_FILENAME, "r") as f:
        dataset = json.load(f)
    if not dataset:
        print("Error: Dataset is empty.")
        exit()
    print(f"Loaded {len(dataset)} entries.")
except FileNotFoundError:
    print(f"Error: Dataset file '{DATASET_FILENAME}' not found.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {DATASET_FILENAME}.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading data: {e}")
    exit()


# --- Preprocess data ---
X = []
y = []
num_invalid_entries = 0

print("Preprocessing data...")
for i, entry in enumerate(dataset):
    try:
        # ***FIX: Access the list of strokes***
        strokes = entry["strokes"]
        if not strokes: # Skip if a trial somehow has no strokes
             print(f"Warning: Entry {i} has no strokes. Skipping.")
             num_invalid_entries += 1
             continue

        # Combine all points from all strokes into a single list for this trial
        all_points_list = []
        for stroke in strokes:
            # Ensure stroke is not empty and points are valid lists/tuples
            if stroke and all(isinstance(p, (list, tuple)) and len(p) == 3 for p in stroke):
                 all_points_list.extend(stroke)
            else:
                print(f"Warning: Invalid stroke format in entry {i}. Skipping stroke.")


        if not all_points_list: # Skip if all strokes were invalid or empty
             print(f"Warning: Entry {i} has no valid points after processing strokes. Skipping.")
             num_invalid_entries += 1
             continue

        all_points_array = np.array(all_points_list, dtype=np.float32)

        # Check for NaNs in the combined array *before* normalization
        if np.isnan(all_points_array).any():
             print(f"Warning: Entry {i} contains NaN values in points. Skipping.")
             num_invalid_entries += 1
             continue

        # Normalize points (X, Y coordinates)
        min_coords = np.min(all_points_array[:, :2], axis=0)
        max_coords = np.max(all_points_array[:, :2], axis=0)
        range_coords = max_coords - min_coords
        # Avoid division by zero if all X or Y are the same
        range_coords[range_coords < 1e-6] = 1.0
        normalized_coords = (all_points_array[:, :2] - min_coords) / range_coords

        # Normalize pressure
        max_pressure = np.max(all_points_array[:, 2])
        normalized_pressure = all_points_array[:, 2] / (max_pressure + 1e-6) # Add epsilon

        # Combine normalized X, Y, Pressure
        processed_points = np.hstack((normalized_coords, normalized_pressure[:, np.newaxis]))

        # Truncate or pad the combined points sequence
        num_points = processed_points.shape[0]
        if num_points > MAX_TOTAL_POINTS:
            final_points = processed_points[:MAX_TOTAL_POINTS]
        else:
            # Pad with zeros (or another value like -1 if preferred)
            padding = np.zeros((MAX_TOTAL_POINTS - num_points, 3), dtype=np.float32)
            final_points = np.vstack((processed_points, padding))

        # --- Process Features ---
        # ***FIX: Access the list of feature dicts***
        features_list = entry.get("features", []) # Use .get for safety

        # Calculate average features across all strokes in the trial
        valid_linearities = [f["avg_linearity"] for f in features_list if f and f.get("avg_linearity") is not None]
        valid_curvatures = [f["avg_curvature"] for f in features_list if f and f.get("avg_curvature") is not None]

        avg_linearity = np.mean(valid_linearities) if valid_linearities else 0.0 # Default to 0 if no valid features
        avg_curvature = np.mean(valid_curvatures) if valid_curvatures else 0.0

        # Flatten points and append *average* features
        # Ensure features are float32
        feature_vector = np.concatenate([
             final_points.flatten(),
             np.array([avg_linearity, avg_curvature], dtype=np.float32)
        ])

        # Check for NaNs in the final feature vector
        if np.isnan(feature_vector).any():
            print(f"Warning: Entry {i} resulted in NaN features after processing. Skipping.")
            # print(f"  Label: {entry.get('label')}, AvgLin: {avg_linearity}, AvgCurv: {avg_curvature}") # Debug print
            num_invalid_entries += 1
            continue

        X.append(feature_vector)
        y.append(entry["label"])

    except KeyError as e:
        print(f"Warning: Missing key {e} in dataset entry {i}. Skipping entry.")
        num_invalid_entries += 1
    except Exception as e:
        print(f"Error processing entry {i} (Label: {entry.get('label', 'N/A')}): {e}")
        # traceback.print_exc() # Uncomment for detailed traceback
        num_invalid_entries += 1


print(f"Preprocessing finished. Valid entries: {len(X)}, Skipped entries: {num_invalid_entries}")

if not X:
    print("Error: No valid data entries found after preprocessing. Cannot train.")
    exit()

# --- Encode labels ---
print("Encoding labels...")
le = LabelEncoder()
# Use try-except for robustness if y is empty
try:
    y_encoded = le.fit_transform(y)
    print(f"Found {len(le.classes_)} classes: {le.classes_}")
except ValueError as e:
     print(f"Error during label encoding (maybe no valid data?): {e}")
     exit()

X = np.array(X, dtype=np.float32)
y_encoded = np.array(y_encoded, dtype=np.int64)


# --- Split dataset ---
print("Splitting dataset...")
# Check if dataset is large enough for splitting
if len(X) < 5: # Need at least a few samples to split reasonably
     print("Warning: Dataset too small for train/test split. Using all data for training.")
     X_train, X_test, y_train, y_test = X, X, y_encoded, y_encoded
elif np.min(np.bincount(y_encoded)) < 2: # Check if any class has < 2 samples for stratification
     print("Warning: Some classes have only one sample. Using non-stratified split.")
     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=None)
else:
     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

if len(X_train) == 0 or len(X_test) == 0:
    print("Error: Train or test set is empty after splitting. Need more data.")
    exit()


# --- Convert to tensors ---
print("Converting data to PyTorch tensors...")
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# --- Define neural network ---
class KanjiClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(KanjiClassifier, self).__init__()
        # Simple MLP (Multi-Layer Perceptron)
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3) # Add dropout for regularization
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes) # Output layer

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x) # No activation here, CrossEntropyLoss applies LogSoftmax
        return x

# --- Initialize model, loss, optimizer ---
input_size = X.shape[1]  # Recalculate based on actual data shape
num_classes = len(le.classes_)
model = KanjiClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Common starting learning rate

print(f"Model initialized. Input size: {input_size}, Num classes: {num_classes}")

# --- Train model ---
num_epochs = 200 # Increased epochs might be needed
print(f"Starting training for {num_epochs} epochs...")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


for epoch in range(num_epochs):
    model.train() # Set model to training mode
    optimizer.zero_grad() # Clear gradients
    outputs = model(X_train) # Forward pass
    loss = criterion(outputs, y_train) # Calculate loss
    loss.backward() # Backward pass
    optimizer.step() # Update weights

    # Evaluate on test set every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation for evaluation
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            correct = (predicted == y_test).sum().item()
            total = y_test.size(0)
            accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

# --- Save model ---
try:
    # Move model back to CPU before saving state_dict for better compatibility
    model.to('cpu')
    torch.save(model.state_dict(), MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")
except Exception as e:
    print(f"Error saving model: {e}")
