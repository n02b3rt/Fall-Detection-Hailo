import os
import sys
import glob
import subprocess
import json
import time

def run_analysis(test_dir=None):
    if test_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(script_dir, "test_cases")

    print(f"Starting analysis of: {test_dir}/*.avi")
    
    video_files = glob.glob(os.path.join(test_dir, "*.avi"))
    if not video_files:
        print("No .avi files found in test_cases/")
        return

    results = []
    
    script_path = os.path.join(os.path.dirname(__file__), "fall_detection.py")
    
    for video in video_files:
        print(f"\nProcessing: {os.path.basename(video)}...")
        
        json_output = video.replace(".avi", ".json")
        
        cmd = [
            sys.executable,
            script_path,
            "--input", video,
            "--hef-path", "/usr/share/hailo-models/yolov8s_pose_h10.hef",
            "--width", "640", "--height", "480",
            "--disable-sync",
            "--headless",
            "--json-output", json_output
        ]
        
        try:
            start_time = time.time()
            # Run with capture_output to inspect failure
            # Timeout: 5 minutes max per video (should be much faster)
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            duration = time.time() - start_time
            
            # Read result
            if os.path.exists(json_output):
                with open(json_output, 'r') as f:
                    data = json.load(f)
                    data['filename'] = os.path.basename(video)
                    data['duration_process'] = duration
                    results.append(data)
                print(f"  -> Done in {duration:.1f}s. Max Score: {data['max_score']:.2f}, Alarms: {data['alarm_count']}")
            else:
                print("  -> Failed to generate JSON report. Output:")
                print(result.stdout)
                print(result.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"  -> Error running execution: {e}")
            print("STDERR:", e.stderr)
            print("STDOUT:", e.stdout)
        except Exception as e:
            print(f"  -> Unexpected error: {e}")

    # Generate Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Filename':<30} | {'Max Score':<10} | {'Alarms':<8} | {'Result':<10}")
    print("-" * 65)
    
    for res in results:
        # Determine PASS/FAIL logic?
        # If filename contains "adl" (Activity of Daily Living), alarm should be 0.
        # If filename contains "fall", alarm should be > 0.
        
        note = ""
        fname = res['filename'].lower()
        alarms = res['alarm_count']
        
        if "adl" in fname or "safe" in fname:
            result = "PASS" if alarms == 0 else "FALSE POS"
        elif "fall" in fname:
            result = "PASS" if alarms > 0 else "FALSE NEG"
        else:
            result = "INFO"
            
        print(f"{res['filename']:<30} | {res['max_score']:<10.2f} | {alarms:<8} | {result:<10}")
    
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_analysis(sys.argv[1])
    else:
        run_analysis()
