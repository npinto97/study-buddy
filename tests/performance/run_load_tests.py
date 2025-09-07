import subprocess
import os
import sys
from datetime import datetime

RESULTS_DIR = "./load_test_reports"
LOCUST_FILE = "locust_test.py"
os.makedirs(RESULTS_DIR, exist_ok=True)

API_HOST = "http://127.0.0.1:8000"


SCENARIOS = {
    "light": {
        "users": 3, 
        "spawn_rate": 1, 
        "duration": "60s",
        "class": "LightLoad",
        "description": "Light load with longer wait times"
    },
    "medium": {
        "users": 6, 
        "spawn_rate": 2, 
        "duration": "90s",
        "class": "MediumLoad",
        "description": "Medium load with balanced request patterns"
    },
    "heavy": {
        "users": 10, 
        "spawn_rate": 3, 
        "duration": "120s",
        "class": "HeavyLoad",
        "description": "Heavy load with frequent requests"
    },
    "stress": {
        "users": 15, 
        "spawn_rate": 5, 
        "duration": "180s",
        "class": "StudyBuddyLoadTest",
        "description": "Stress test with maximum load"
    }
}


def check_api_health():
    """Check if API is running before starting tests"""
    try:
        import requests
        response = requests.get(f"{API_HOST}/health", timeout=5)
        if response.status_code == 200:
            print("API is running and healthy")
            return True
        else:
            print(f"API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Cannot reach API: {e}")
        return False


def run_locust_scenario(name, config):
    """Run a single locust scenario"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_prefix = os.path.join(RESULTS_DIR, f"{config['class']}_{name}_{timestamp}")
    
    print(f"\n{'='*60}")
    print(f"Running {name.upper()} scenario:")
    print(f"  Description: {config['description']}")
    print(f"  Users: {config['users']}, Spawn Rate: {config['spawn_rate']}/s")
    print(f"  Duration: {config['duration']}, Class: {config['class']}")
    print(f"{'='*60}")
    
    cmd = [
        "locust",
        "-f", LOCUST_FILE,
        "--headless",
        "--host", API_HOST,
        "-u", str(config["users"]),
        "-r", str(config["spawn_rate"]),
        "--run-time", config["duration"],
        "--class-picker", config["class"],
        "--csv", csv_prefix,
        "--html", f"{csv_prefix}.html"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"{name} scenario completed successfully")
            print(f"  Results saved to: {csv_prefix}.*")
        else:
            print(f"{name} scenario failed")
            print(f"  Error: {result.stderr}")
            
        return csv_prefix if result.returncode == 0 else None
        
    except FileNotFoundError:
        print("Locust not found. Install with: pip install locust")
        return None
    except Exception as e:
        print(f"Error running {name}: {e}")
        return None


def run_specific_scenario(scenario_name):
    """Run a specific scenario"""
    if scenario_name not in SCENARIOS:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available scenarios: {list(SCENARIOS.keys())}")
        return False
    
    if not check_api_health():
        return False
    
    result = run_locust_scenario(scenario_name, SCENARIOS[scenario_name])
    return result is not None


def run_all_scenarios():
    """Run all predefined scenarios"""
    if not check_api_health():
        return
    
    results = {}
    successful = 0
    total = len(SCENARIOS)
    
    print(f"\nStarting load tests for {total} scenarios...")
    
    for name, config in SCENARIOS.items():
        result = run_locust_scenario(name, config)
        results[name] = result
        if result:
            successful += 1
    
    print(f"\n{'='*60}")
    print("LOAD TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total scenarios: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    
    print("\nResults:")
    for name, result in results.items():
        status = "SUCCESS" if result else "FAILED"
        print(f"  {name.ljust(10)} - {status}")
        if result:
            print(f"    Files: {result}.*")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        success = run_specific_scenario(scenario)
        sys.exit(0 if success else 1)
    else:
        run_all_scenarios()