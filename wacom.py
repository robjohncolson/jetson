python
import numpy as np
from numba import cuda
import evdev
import asyncio
import time
from datetime import datetime

# CUDA kernel to analyze stroke linearity (detects if points form a line)
@cuda.jit
def analyze_stroke_linearity(points, output):
    idx = cuda.grid(1)
    if idx < points.shape[0] - 2:
        # Compute vector differences between consecutive points
        dx1 = points[idx + 1, 0] - points[idx, 0]
        dy1 = points[idx + 1, 1] - points[idx, 1]
        dx2 = points[idx + 2, 0] - points[idx + 1, 0]
        dy2 = points[idx + 2, 1] - points[idx + 1, 1]
        # Cross product to measure collinearity
        cross = abs(dx1 * dy2 - dy1 * dx2)
        output[idx] = cross  # Small cross product = more linear

# CUDA kernel to estimate curvature
#  (detects circular patterns)
@cuda.jit
def analyze_stroke_curvature(points, output):
    idx = cuda.grid(1)
    if idx < points.shape[0] - 2:
        # Compute approximate curvature using three points
        x1, y1 = points[idx, 0], points[idx, 1]
        x2, y2 = points[idx + 1, 0], points[idx + 1, 1]
        x3, y3 = points[idx + 2, 0], points[idx + 2, 1]
        # Triangle area for curvature estimation
        area = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
        output[idx] = area  # Larger area = less circular

async def capture_wacom_input(device_path):
    device = evdev.InputDevice(device_path)
    print(f"Listening on {device.name}")
    
    points = []  # Store (x, y, pressure) for each stroke
    stroke_active = False
    output_file = open("stroke_analysis.txt", "a")
    
    async for event in device.async_read_loop():
        if event.type == evdev.ecodes.EV_ABS:
            # Capture pen position and pressure
            if event.code == evdev.ecodes.ABS_X:
                x = event.value
            elif event.code == evdev.ecodes.ABS_Y:
                y = event.value
            elif event.code == evdev.ecodes.ABS_PRESSURE:
                pressure = event.value
                if pressure > 0 and not stroke_active:
                    stroke_active = True
                    points = []
                elif pressure == 0 and stroke_active:
                    stroke_active = False
                    # Process stroke with CUDA
                    if len(points) >= 3:
                        points_array = np.array(points, dtype=np.float32)
                        d_points = cuda.to_device(points_array)
                        output_linearity = cuda.device_array(len(points) - 2, dtype=np.float32)
                        output_curvature = cuda.device_array(len(points) - 2, dtype=np.float32)
                        
                        # Launch CUDA kernels
                        threadsperblock = 256
                        blockspergrid = (len(points) - 2 + threadsperblock - 1) // threadsperblock
                        analyze_stroke_linearity[blockspergrid, threadsperblock](d_points, output_linearity)
                        analyze_stroke_curvature[blockspergrid, threadsperblock](d_points, output_curvature)
                        
                        # Copy results back
                        linearity = output_linearity.copy_to_host()
                        curvature = output_curvature.copy_to_host()
                        
                        # Analyze results
                        avg_linearity = np.mean(linearity)
                        avg_curvature = np.mean(curvature)
                        shape = "Unknown"
                        if avg_linearity < 1000:
                            shape = "Line"
                        elif avg_curvature > 5000:
                            shape = "Circle"
                        
                        # Save to file
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        output_file.write(f"{timestamp} - Shape: {shape}, Points: {len(points)}, Avg Linearity: {avg_linearity:.2f}, Avg Curvature: {avg_curvature:.2f}\n")
                        output_file.flush()
                        print(f"Detected {shape} with {len(points)} points")
            
            if stroke_active:
                points.append([x, y, pressure])
    
    output_file.close()

def main():
    # Replace with your Wacom event device (e.g., /dev/input/event2)
    device_path = "/dev/input/event2"  # Update based on 'cat /proc/bus/input/devices'
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(capture_wacom_input(device_path))
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        loop.close()

if __name__ == "__main__":
    main()
