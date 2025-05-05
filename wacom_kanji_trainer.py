# wacom_kanji_trainer.py
# Captures Wacom pen input (multi-stroke), processes with CUDA,
# saves labeled data for AI training (Kanji/shape recognition).
# Optimized for reliability without visualization.
# FIX: Numba TypingError, SyntaxError on finally in main.

import numpy as np
from numba import cuda
import evdev
import asyncio
import time
from datetime import datetime
import json
import os
import platform # Import platform module
import math     # Import standard math module at the top level
import signal   # Import signal module
import traceback # Import traceback for better error reporting

# --- Configuration ---
PEN_DEVICE_NAME = "Wacom Intuos S Pen" # Name to find the event device
OUTPUT_FILENAME = "kanji_dataset.json"
TRIALS_PER_KANJI = 10
PAUSE_DURATION_SECONDS = 3.0  # Time of inactivity to separate trials
MIN_POINTS_PER_STROKE = 10   # Ignore very short strokes

# --- CUDA Kernels (for feature extraction) ---
@cuda.jit
def analyze_stroke_linearity(points_xy_contiguous, output): # Takes contiguous XY array
    """Calculates cross product for consecutive segments to measure linearity."""
    # Use math functions directly (available in Numba CUDA context)
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
def analyze_stroke_curvature(points_xy_contiguous, output): # Takes contiguous XY array
    """Estimates curvature based on the area of triangles formed by consecutive points."""
    idx = cuda.grid(1)
    if idx < points_xy_contiguous.shape[0] - 2:
        x1, y1 = points_xy_contiguous[idx, 0], points_xy_contiguous[idx, 1]
        x2, y2 = points_xy_contiguous[idx + 1, 0], points_xy_contiguous[idx + 1, 1]
        x3, y3 = points_xy_contiguous[idx + 2, 0], points_xy_contiguous[idx + 2, 1]
        area = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
        output[idx] = area

def get_pen_device_path(device_name):
    """Finds the event device path for the specified Wacom pen name."""
    pen_device_path = None
    try:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for device in devices:
            if device_name.lower() in device.name.lower():
                print(f"Found pen device: {device.path} ({device.name})")
                pen_device_path = device.path
                # Don't close here, let the caller manage the device lifetime
                # device.close()
                break # Found the first matching device
    except Exception as e:
        print(f"Error listing input devices: {e}")
        return None

    if not pen_device_path:
        # Fallback search if primary name fails
        print(f"Wacom pen device '{device_name}' not found by name. Attempting common event paths...")
        for i in range(20): # Check event0 to event19
             potential_path = f"/dev/input/event{i}"
             if os.path.exists(potential_path):
                 try:
                     dev = evdev.InputDevice(potential_path)
                     if PEN_DEVICE_NAME.lower() in dev.name.lower():
                         print(f"Found pen device via fallback: {potential_path}")
                         pen_device_path = potential_path
                         dev.close() # Close after checking name
                         break
                     dev.close()
                 except (OSError, PermissionError, Exception): # Catch more errors during probing
                     pass # Ignore errors for non-Wacom or inaccessible devices
        if not pen_device_path:
            print("Could not find Wacom pen device automatically or via fallback.")
            return None

    return pen_device_path


def load_existing_data(filename):
    """Loads existing data from the JSON file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                # Handle empty file case
                content = f.read()
                if not content:
                    return []
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

def process_stroke_features(stroke_points):
    """Processes a single stroke to calculate linearity and curvature using CUDA."""
    if len(stroke_points) < 3:
        return {"avg_linearity": None, "avg_curvature": None}

    points_xy = np.array(stroke_points, dtype=np.float32)[:, :2]
    points_xy_contiguous = np.ascontiguousarray(points_xy)
    valid_indices = ~np.isnan(points_xy_contiguous).any(axis=1)
    points_xy_contiguous_valid = points_xy_contiguous[valid_indices]

    if points_xy_contiguous_valid.shape[0] < 3:
         return {"avg_linearity": None, "avg_curvature": None}

    try:
        d_points = cuda.to_device(points_xy_contiguous_valid)
        num_segments = points_xy_contiguous_valid.shape[0] - 2
        output_linearity = cuda.device_array(num_segments, dtype=np.float32)
        output_curvature = cuda.device_array(num_segments, dtype=np.float32)

        threadsperblock = 128
        blockspergrid = (num_segments + threadsperblock - 1) // threadsperblock

        if blockspergrid <= 0:
             return {"avg_linearity": None, "avg_curvature": None}

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
        print(f"CUDA processing error: {e}")
        # traceback.print_exc() # Uncomment for detailed debug
        return {"avg_linearity": None, "avg_curvature": None}

async def capture_wacom_input(pen_device_path, label, all_data):
    """Captures Wacom strokes, detects pauses, saves labeled trials."""
    pen_device = None
    try:
        pen_device = evdev.InputDevice(pen_device_path)
        print(f"Listening on {pen_device.name}")
    except Exception as e:
        print(f"Error opening device {pen_device_path}: {e}")
        return # Exit if device cannot be opened

    current_stroke = []
    trial_strokes = []
    stroke_active = False
    last_event_time = time.monotonic()
    pause_start_time = None
    trials_collected = 0
    x, y, pressure = None, None, 0

    print(f"Training for '{label}'. Draw {TRIALS_PER_KANJI} trials. Start drawing; {PAUSE_DURATION_SECONDS}-second pause separates trials.")

    exit_flag = asyncio.Event()
    current_task = asyncio.current_task() # Get the current task

    def signal_handler(sig, frame):
        print("\nCtrl+C detected, setting exit flag...")
        exit_flag.set()
        if current_task and not current_task.done():
             # Request cancellation of the capture_wacom_input task
             current_task.cancel()

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while trials_collected < TRIALS_PER_KANJI:
            if exit_flag.is_set():
                 print("Exit flag set, breaking capture loop.")
                 break

            event_processed = False
            try:
                # Read events with a short timeout
                event = await asyncio.wait_for(pen_device.async_read_one(), timeout=0.05)
            except asyncio.TimeoutError:
                event = None
            except asyncio.CancelledError:
                 print("Capture task was cancelled.")
                 break # Exit loop if task cancelled
            except OSError as e:
                 if e.errno == 19: # ENODEV
                     print(f"Warning: Device {pen_device_path} unavailable (errno 19). Retrying connection...")
                     await asyncio.sleep(1.0)
                     try:
                         pen_device = evdev.InputDevice(pen_device_path) # Try reopening
                         print(f"Reconnected to {pen_device.name}")
                     except Exception as reopen_e:
                         print(f"Failed to reconnect: {reopen_e}. Exiting capture.")
                         break
                     continue
                 else:
                     print(f"Warning: Read error on {pen_device_path}: {e}. Retrying...")
                     await asyncio.sleep(0.5)
                 continue
            except Exception as e:
                 print(f"Unexpected error during read: {e}")
                 break

            if event:
                event_processed = True
                last_event_time = time.monotonic()
                pause_start_time = None # Reset pause timer on any activity

                if event.type == evdev.ecodes.EV_ABS:
                    # Update coordinates/pressure based on the event
                    if event.code == evdev.ecodes.ABS_X:
                        x = event.value
                    elif event.code == evdev.ecodes.ABS_Y:
                        y = event.value
                    elif event.code == evdev.ecodes.ABS_PRESSURE:
                        new_pressure = event.value
                        # Check for pressure change to detect pen down/up
                        if new_pressure > 0 and pressure == 0: # Pen down
                            stroke_active = True
                            current_stroke = []
                            # Record the first point immediately if coords are known
                            if x is not None and y is not None:
                                current_stroke.append([float(x), float(y), float(new_pressure)])
                        elif new_pressure == 0 and pressure > 0: # Pen up
                            stroke_active = False
                            if len(current_stroke) >= MIN_POINTS_PER_STROKE:
                                print(f"  Stroke {len(trial_strokes) + 1} captured with {len(current_stroke)} points")
                                trial_strokes.append(list(current_stroke)) # Append copy
                            # Reset current_stroke only AFTER potentially adding it
                            current_stroke = []
                        pressure = new_pressure # Update pressure state *after* checking transition

                # Append point if stroke is active and coordinates are valid
                # Make sure pressure is also updated from the event stream
                if stroke_active and x is not None and y is not None and pressure is not None:
                    if pressure > 0:
                         # Use the most recently updated x, y, pressure
                        current_stroke.append([float(x), float(y), float(pressure)])

            # --- Pause Detection Logic ---
            current_time = time.monotonic()

            # Start pause timer only if pen is up AND there are strokes collected for the current trial
            if not stroke_active and trial_strokes and pause_start_time is None:
                 # Use the time of the last event as the pause start
                 pause_start_time = last_event_time

            # If we are in a potential pause state, check duration
            if pause_start_time is not None:
                pause_elapsed = current_time - pause_start_time
                remaining_time = PAUSE_DURATION_SECONDS - pause_elapsed

                # Print countdown (update less frequently)
                # Check if remaining time has crossed an integer second boundary
                if remaining_time >= 0 and (int(remaining_time * 10) % 5 == 0): # Update roughly every 0.5s
                     print(f"Inactive: {remaining_time:.1f}s      ", end='\r')

                # Check if pause duration is met
                if pause_elapsed >= PAUSE_DURATION_SECONDS:
                    print("                                ", end='\r') # Clear countdown line
                    if trial_strokes: # Ensure we have strokes to save
                        trials_collected += 1
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                        print(f"Processing Trial {trials_collected} (Strokes: {len(trial_strokes)})...")
                        stroke_features = []
                        for i, stroke in enumerate(trial_strokes):
                             # print(f"  Processing Stroke {i+1}/{len(trial_strokes)}...") # Optional: more verbose
                             features = process_stroke_features(stroke)
                             stroke_features.append(features)
                             # print(f"    Features: {features}") # Optional: more verbose

                        trial_data = {
                            "timestamp": timestamp,
                            "label": label,
                            "strokes": trial_strokes,
                            "features": stroke_features
                        }
                        all_data.append(trial_data)
                        # Save data *immediately* after processing each trial
                        save_data(OUTPUT_FILENAME, all_data)
                        print(f"Trial {trials_collected} completed with {len(trial_strokes)} strokes. Saved.")

                        trial_strokes = [] # Reset for the next trial
                        pause_start_time = None

                        if trials_collected >= TRIALS_PER_KANJI:
                            break # Exit the while loop for this label

            # Prevent busy-waiting if no events read this cycle
            if not event_processed:
                await asyncio.sleep(0.02) # Yield control briefly

    except asyncio.CancelledError:
        print("Capture task was cancelled (likely by Ctrl+C).")
    except Exception as e:
        print(f"Error during capture loop: {e}")
        traceback.print_exc()
    finally:
        # Restore original signal handler when capture function exits
        signal.signal(signal.SIGINT, original_sigint_handler)
        if pen_device and pen_device.fd != -1:
             try:
                 pen_device.close()
             except Exception as e:
                 print(f"Note: Error closing device: {e}")
        print("Finished capture loop.")


def main():
    pen_device_path = get_pen_device_path(PEN_DEVICE_NAME)
    if not pen_device_path:
        return

    if not os.access(pen_device_path, os.R_OK):
        print(f"Error: Read permission denied for {pen_device_path}.")
        print(f"Try: sudo usermod -a -G input {os.getlogin()} (then log out/in)")
        return

    all_data = load_existing_data(OUTPUT_FILENAME)
    print(f"Loaded {len(all_data)} existing entries.")

    # Get the current running loop or create a new one if none exists
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    exit_main_loop = False
    while not exit_main_loop:
        try:
            label = input(f"\nEnter the Kanji/shape to train (e.g., '大', 'circle') or 'quit' to exit: ").strip()
            if not label:
                print("Label cannot be empty.")
                continue
            if label.lower() == 'quit':
                exit_main_loop = True # Signal to exit the outer loop
                continue # Skip to the next iteration to exit loop

            # Run the capture task for the current label
            capture_task = loop.create_task(capture_wacom_input(pen_device_path, label, all_data))
            loop.run_until_complete(capture_task)

            # Check if task was cancelled (e.g., by Ctrl+C inside capture)
            if capture_task.cancelled():
                print("Capture was cancelled. Exiting.")
                exit_main_loop = True
            else:
                # Check result if needed, though capture_wacom_input doesn't return anything specific
                try:
                     capture_task.result() # Check for exceptions raised within the task
                     print(f"\nSession for '{label}' finished.")
                except asyncio.CancelledError:
                     print("Capture task was cancelled during completion check.")
                     exit_main_loop = True # Exit if cancelled here too

        except KeyboardInterrupt:
            print("\nProgram terminated by user (main loop).")
            exit_main_loop = True # Signal to exit the outer loop
        except Exception as e:
            print(f"\nAn unexpected error occurred in main loop: {e}")
            traceback.print_exc()
            exit_main_loop = True # Exit on unexpected errors

    # Clean up the loop after the while loop exits
    try:
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
             # Gently cancel remaining tasks
             tasks = asyncio.all_tasks(loop=loop)
             for task in tasks:
                  task.cancel()
             # Allow tasks to finish cancelling
             loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
             loop.close()
    except Exception as e:
        print(f"Error closing asyncio loop: {e}")

    print("Exiting program.")


if __name__ == "__main__":
    main()
