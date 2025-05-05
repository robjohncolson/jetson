# kanji_trainer_predict.py
# Loads the trained model and predicts Kanji drawn on the Wacom tablet.
# FIX: Handle multi-stroke data correctly, match preprocessing & input size from training.
# FIX: Add missing get_pen_device_path, load_existing_data, save_data function definitions.

import numpy as np
from numba import cuda
import evdev
import asyncio
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime
import os
import platform
import math
import signal
import traceback

# --- Configuration ---
PEN_DEVICE_NAME = "Wacom Intuos S Pen"
DATASET_FILENAME = "kanji_dataset.json"
MODEL_FILENAME = "kanji_classifier.pth"
MAX_TOTAL_POINTS = 600  # *** MUST MATCH THE VALUE USED IN kanji_trainer_model.py ***
PAUSE_DURATION_SECONDS = 3.0
MIN_POINTS_PER_STROKE = 10

# --- Helper Functions (Copied from data collection script) ---
def get_pen_device_path(device_name):
    """Finds the event device path for the specified Wacom pen name."""
    pen_device_path = None
    try:
        print("Searching for input devices...")
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        print(f"Found {len(devices)} devices. Looking for '{device_name}'...")
        found_dev = None
        for device in devices:
            try:
                if device_name.lower() in device.name.lower():
                    print(f"Found matching device: {device.path} ({device.name})")
                    pen_device_path = device.path
                    found_dev = device
                    break
            except Exception as e:
                print(f"Warning: Could not read info for device {device.path}: {e}")
            finally:
                 if found_dev is None and 'device' in locals() and device.fd != -1:
                     try: device.close()
                     except: pass
        if found_dev and found_dev.fd != -1:
             try: found_dev.close()
             except: pass
    except Exception as e:
        print(f"Error listing input devices: {e}")
        return None

    if not pen_device_path:
        print(f"Wacom pen device '{device_name}' not found by name. Attempting common event paths...")
        for i in range(20):
             potential_path = f"/dev/input/event{i}"
             if os.path.exists(potential_path):
                 try:
                     dev = evdev.InputDevice(potential_path)
                     dev_name = dev.name
                     dev.close()
                     if PEN_DEVICE_NAME.lower() in dev_name.lower():
                         print(f"Found pen device via fallback: {potential_path} ({dev_name})")
                         pen_device_path = potential_path
                         break
                 except (OSError, PermissionError, Exception):
                     try:
                         if 'dev' in locals() and dev.fd != -1: dev.close()
                     except: pass
        if not pen_device_path:
            print("Could not find Wacom pen device automatically or via fallback.")
            return None
    return pen_device_path

def load_existing_data(filename):
    """Loads existing data from the JSON file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                content = f.read()
                if not content: return []
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filename}. Starting fresh.")
            return []
        except Exception as e:
            print(f"Error loading data: {e}. Starting fresh.")
            return []
    return []

def save_data(filename, data):
    """Saves the updated data to the JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving data: {e}")

# --- CUDA Kernels (Must be identical to training script) ---
@cuda.jit
def analyze_stroke_linearity(points_xy_contiguous, output):
    # Use math functions directly
    idx = cuda.grid(1)
    if idx < points_xy_contiguous.shape[0] - 2:
        dx1 = points_xy_contiguous[idx + 1, 0] - points_xy_contiguous[idx, 0]
        dy1 = points_xy_contiguous[idx + 1, 1] - points_xy_contiguous[idx, 1]
        dx2 = points_xy_contiguous[idx + 2, 0] - points_xy_contiguous[idx + 1, 0]
        dy2 = points_xy_contiguous[idx + 2, 1] - points_xy_contiguous[idx + 1, 1]
        len1_sq = dx1*dx1 + dy1*dy1
        len2_sq = dx2*dx2 + dy2*dy2
        if len1_sq > 1e-9 and len2_sq > 1e-9:
             cross = abs(dx1 * dy2 - dy1 * dx2)
             denominator = math.sqrt(len1_sq * len2_sq)
             output[idx] = cross / denominator if denominator > 1e-9 else 0.0
        else:
             output[idx] = 0.0

@cuda.jit
def analyze_stroke_curvature(points_xy_contiguous, output):
    idx = cuda.grid(1)
    if idx < points_xy_contiguous.shape[0] - 2:
        x1, y1 = points_xy_contiguous[idx, 0], points_xy_contiguous[idx, 1]
        x2, y2 = points_xy_contiguous[idx + 1, 0], points_xy_contiguous[idx + 1, 1]
        x3, y3 = points_xy_contiguous[idx + 2, 0], points_xy_contiguous[idx + 2, 1]
        area = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
        output[idx] = area

# --- Feature Processing Function (Must be identical to training script) ---
def process_stroke_features(stroke_points):
    if len(stroke_points) < 3: return {"avg_linearity": None, "avg_curvature": None}
    points_xy = np.array(stroke_points, dtype=np.float32)[:, :2]
    points_xy_contiguous = np.ascontiguousarray(points_xy)
    valid_indices = ~np.isnan(points_xy_contiguous).any(axis=1)
    points_xy_contiguous_valid = points_xy_contiguous[valid_indices]
    if points_xy_contiguous_valid.shape[0] < 3: return {"avg_linearity": None, "avg_curvature": None}
    try:
        d_points = cuda.to_device(points_xy_contiguous_valid)
        num_segments = points_xy_contiguous_valid.shape[0] - 2
        output_linearity = cuda.device_array(num_segments, dtype=np.float32)
        output_curvature = cuda.device_array(num_segments, dtype=np.float32)
        threadsperblock = 128
        blockspergrid = (num_segments + threadsperblock - 1) // threadsperblock
        if blockspergrid <= 0: return {"avg_linearity": None, "avg_curvature": None}
        analyze_stroke_linearity[blockspergrid, threadsperblock](d_points, output_linearity)
        cuda.synchronize()
        analyze_stroke_curvature[blockspergrid, threadsperblock](d_points, output_curvature)
        cuda.synchronize()
        linearity_results = output_linearity.copy_to_host()
        curvature_results = output_curvature.copy_to_host()
        avg_linearity = np.nanmean(linearity_results) if not np.all(np.isnan(linearity_results)) else None
        avg_curvature = np.nanmean(curvature_results) if not np.all(np.isnan(curvature_results)) else None
        return {"avg_linearity": float(avg_linearity) if avg_linearity is not None and not np.isnan(avg_linearity) else None,
                "avg_curvature": float(avg_curvature) if avg_curvature is not None and not np.isnan(avg_curvature) else None}
    except Exception as e:
        print(f"CUDA processing error during prediction feature extraction: {e}")
        return {"avg_linearity": None, "avg_curvature": None}


# --- Define neural network (Must be identical to training script) ---
class KanjiClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(KanjiClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# --- Preprocessing Function (Must be identical to training script) ---
def preprocess_trial_for_prediction(trial_strokes):
    """Prepares a multi-stroke trial for model prediction."""
    if not trial_strokes: return None, None
    all_points_list = []
    for stroke in trial_strokes:
        if stroke and all(isinstance(p, (list, tuple)) and len(p) == 3 for p in stroke):
             all_points_list.extend(stroke)
    if not all_points_list: return None, None
    all_points_array = np.array(all_points_list, dtype=np.float32)
    if np.isnan(all_points_array).any(): return None, None

    # Normalize points
    min_coords = np.min(all_points_array[:, :2], axis=0)
    max_coords = np.max(all_points_array[:, :2], axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords < 1e-6] = 1.0
    normalized_coords = (all_points_array[:, :2] - min_coords) / range_coords
    max_pressure = np.max(all_points_array[:, 2])
    normalized_pressure = all_points_array[:, 2] / (max_pressure + 1e-6)
    processed_points = np.hstack((normalized_coords, normalized_pressure[:, np.newaxis]))

    # Truncate or pad
    num_points = processed_points.shape[0]
    if num_points > MAX_TOTAL_POINTS:
        final_points = processed_points[:MAX_TOTAL_POINTS]
    else:
        padding = np.zeros((MAX_TOTAL_POINTS - num_points, 3), dtype=np.float32)
        final_points = np.vstack((processed_points, padding))

    # Calculate average features
    stroke_features_list = [process_stroke_features(stroke) for stroke in trial_strokes]
    valid_linearities = [f["avg_linearity"] for f in stroke_features_list if f and f.get("avg_linearity") is not None]
    valid_curvatures = [f["avg_curvature"] for f in stroke_features_list if f and f.get("avg_curvature") is not None]
    avg_linearity = np.mean(valid_linearities) if valid_linearities else 0.0
    avg_curvature = np.mean(valid_curvatures) if valid_curvatures else 0.0

    # Create feature vector
    feature_vector = np.concatenate([
         final_points.flatten(),
         np.array([avg_linearity, avg_curvature], dtype=np.float32)
    ])
    if np.isnan(feature_vector).any(): return None, None

    features_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
    return features_tensor, trial_strokes


async def capture_and_predict(pen_device_path, model, le, device, all_data):
    """Captures Wacom strokes, detects pauses, predicts label, optionally saves."""
    pen_device = None
    try:
        pen_device = evdev.InputDevice(pen_device_path)
        print(f"Listening on {pen_device.name}")
    except Exception as e:
        print(f"Error opening device {pen_device_path}: {e}")
        return

    current_stroke = []
    trial_strokes = []
    stroke_active = False
    last_event_time = time.monotonic()
    pause_start_time = None
    x, y, pressure = None, None, 0

    print(f"Draw a Kanji/shape. Pause for {PAUSE_DURATION_SECONDS}s to trigger prediction.")

    exit_flag = asyncio.Event()
    current_task = asyncio.current_task()

    def signal_handler(sig, frame):
        print("\nCtrl+C detected, setting exit flag...")
        exit_flag.set()
        if current_task and not current_task.done():
             current_task.cancel()

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while not exit_flag.is_set():
            event_processed = False
            try:
                event = await asyncio.wait_for(pen_device.async_read_one(), timeout=0.05)
            except asyncio.TimeoutError: event = None
            except asyncio.CancelledError: print("Capture task cancelled."); break
            except OSError as e:
                 if e.errno == 19: # ENODEV
                     print(f"Warning: Device {pen_device_path} unavailable. Retrying...")
                     await asyncio.sleep(1.0)
                     try: pen_device = evdev.InputDevice(pen_device_path); print(f"Reconnected.")
                     except Exception as reopen_e: print(f"Reconnect failed: {reopen_e}. Exiting."); break
                     continue
                 else: print(f"Warning: Read error: {e}. Retrying..."); await asyncio.sleep(0.5); continue
            except Exception as e: print(f"Unexpected error during read: {e}"); break

            if event:
                event_processed = True
                last_event_time = time.monotonic()
                pause_start_time = None

                if event.type == evdev.ecodes.EV_ABS:
                    if event.code == evdev.ecodes.ABS_X: x = event.value
                    elif event.code == evdev.ecodes.ABS_Y: y = event.value
                    elif event.code == evdev.ecodes.ABS_PRESSURE:
                        new_pressure = event.value
                        if new_pressure > 0 and pressure == 0: # Pen down
                            stroke_active = True
                            current_stroke = []
                            if x is not None and y is not None: current_stroke.append([float(x), float(y), float(new_pressure)])
                        elif new_pressure == 0 and pressure > 0: # Pen up
                            stroke_active = False
                            if len(current_stroke) >= MIN_POINTS_PER_STROKE:
                                print(f"  Stroke {len(trial_strokes) + 1} captured with {len(current_stroke)} points")
                                trial_strokes.append(list(current_stroke))
                            current_stroke = []
                        pressure = new_pressure

                if stroke_active and x is not None and y is not None and pressure is not None:
                    if pressure > 0: current_stroke.append([float(x), float(y), float(pressure)])

            # --- Pause Detection Logic ---
            current_time = time.monotonic()

            if not stroke_active and trial_strokes and pause_start_time is None:
                 pause_start_time = last_event_time

            if pause_start_time is not None:
                pause_elapsed = current_time - pause_start_time
                remaining_time = PAUSE_DURATION_SECONDS - pause_elapsed
                if int(remaining_time * 10) % 5 == 0 and remaining_time >= 0: print(f"Inactive: {remaining_time:.1f}s      ", end='\r')

                if pause_elapsed >= PAUSE_DURATION_SECONDS:
                    print("Processing for prediction...          ", end='\r')
                    features_tensor, saved_strokes = preprocess_trial_for_prediction(trial_strokes)

                    if features_tensor is not None:
                        features_tensor = features_tensor.to(device)
                        with torch.no_grad():
                            output = model(features_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            confidence, predicted_idx = torch.max(probabilities, 1)
                            predicted_label = le.inverse_transform(predicted_idx.cpu().numpy())[0]

                        print(f"\nPredicted shape/Kanji: {predicted_label} (Confidence: {confidence.item():.2f})")

                        try:
                            correct_label = input("Enter correct label (or press Enter to skip): ").strip()
                            if correct_label:
                                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                print("Calculating features for saving...") # Add feedback
                                stroke_features = [process_stroke_features(stroke) for stroke in saved_strokes]
                                trial_data = {
                                    "timestamp": timestamp, "label": correct_label,
                                    "strokes": saved_strokes, "features": stroke_features
                                }
                                all_data.append(trial_data)
                                save_data(DATASET_FILENAME, all_data) # Use the helper function
                                print(f"Added stroke labeled '{correct_label}' to dataset.")
                                # Re-fit label encoder
                                labels = [entry["label"] for entry in all_data if "label" in entry]
                                le.classes_ = np.array(list(set(labels))) # Update classes directly if possible, or refit
                                # le.fit(labels) # Refitting might change encoding order, be careful
                            else:
                                print("Skipped saving.")
                        except Exception as e: print(f"Error during label input/saving: {e}")
                    else:
                        print("\nCould not process strokes for prediction (invalid data?).")

                    trial_strokes = [] # Reset for the next prediction trial
                    pause_start_time = None
                    print(f"\nReady for next drawing. Draw a Kanji/shape. Pause for {PAUSE_DURATION_SECONDS}s to predict.")

            if not event_processed and not stroke_active and not pause_start_time : await asyncio.sleep(0.02)

    except asyncio.CancelledError: print("Prediction task cancelled (likely by Ctrl+C).")
    except Exception as e: print(f"Error in capture_and_predict: {e}"); traceback.print_exc()
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if pen_device and pen_device.fd != -1:
             try: pen_device.close()
             except Exception as e: print(f"Note: Error closing device: {e}")
        print("Finished prediction loop.")

def main():
    # --- Device Setup ---
    pen_device_path = get_pen_device_path(PEN_DEVICE_NAME)
    if not pen_device_path: return
    if not os.access(pen_device_path, os.R_OK):
        print(f"Error: Read permission denied for {pen_device_path}.")
        print(f"Try: sudo usermod -a -G input {os.getlogin()} (then log out/in)")
        return

    # --- Load Data and Label Encoder ---
    print(f"Loading dataset from {DATASET_FILENAME} to fit LabelEncoder...")
    # ***FIX: Use the helper function***
    all_data = load_existing_data(DATASET_FILENAME)
    if not all_data: print(f"Warning: Dataset '{DATASET_FILENAME}' is empty or invalid."); labels = []
    else: labels = [entry["label"] for entry in all_data if "label" in entry]

    if not labels:
         print("Warning: No labels found in the dataset. Cannot fit LabelEncoder.")
         print("Please collect some labeled data first using the training script.")
         return

    le = LabelEncoder()
    le.fit(labels)
    print(f"Fitted LabelEncoder with classes: {le.classes_}")

    # --- Load Model ---
    print(f"Loading model from {MODEL_FILENAME}...")
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: Model file '{MODEL_FILENAME}' not found. Please train the model first.")
        return

    input_size = MAX_TOTAL_POINTS * 3 + 2
    num_classes = len(le.classes_)
    model = KanjiClassifier(input_size, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError: print(f"Error: Model file '{MODEL_FILENAME}' not found."); return
    except RuntimeError as e: print(f"Error loading model state_dict: {e}"); print("Check MAX_TOTAL_POINTS match."); return
    except Exception as e: print(f"An unexpected error occurred loading the model: {e}"); return

    # --- Start Prediction Loop ---
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Pass all_data so it can be appended to if user corrects label
        loop.run_until_complete(capture_and_predict(pen_device_path, model, le, device, all_data))
    except KeyboardInterrupt: print("\nProgram terminated by user (main).")
    except Exception as e: print(f"\nAn unexpected error occurred in prediction main loop: {e}"); traceback.print_exc()
    finally:
         try:
            if loop.is_running(): loop.stop()
            if not loop.is_closed():
                tasks = asyncio.all_tasks(loop=loop)
                for task in tasks: task.cancel()
                # Use asyncio.wait to handle potentially finished/cancelled tasks
                loop.run_until_complete(asyncio.wait(tasks if tasks else []))
                loop.close()
         except Exception as e: print(f"Error closing asyncio loop: {e}")
         print("Exiting prediction program.")

if __name__ == "__main__":
    main()
