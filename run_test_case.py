import sys
import subprocess
import os

def run_test_case(input_file):
    print(f"Running test case: {input_file}")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)
        
    # Command to run fall_detection.py
    # This assumes we are in the same directory or can find it
    script_path = os.path.join(os.path.dirname(__file__), "fall_detection.py")
    
    cmd = [
        sys.executable,
        script_path,
        "--input", input_file,
        "--hef-path", "/usr/share/hailo-models/yolov8s_pose_h10.hef",
        # Assuming standard resolution or input file res? 
        # Gstreamer usually handles file res automatically if not specified?
        # But GStreamerApp args might require width/height for display sink caps?
        # Or source pipeline?
        # Let's provide standard args consistent with web app default
        "--width", "640", "--height", "480", 
        "--disable-sync"
    ]
    
    # Execute
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except Exception as e:
        print(f"Error running test: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_test_case.py <path_to_video.avi>")
        sys.exit(1)
        
    run_test_case(sys.argv[1])
