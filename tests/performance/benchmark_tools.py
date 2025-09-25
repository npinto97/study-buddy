import os
import json
import time
import requests
from datetime import datetime


API_URL = "http://127.0.0.1:8000"
BENCHMARK_DIR = "./benchmark_reports"
os.makedirs(BENCHMARK_DIR, exist_ok=True)


TEST_PDF_FILE = "./test_files/sample.pdf"
TEST_CSV_FILE = "./test_files/sample_data.csv"


TOOLS = {
    "vector_search": {
        "url": "/test/vector-search",
        "method": "POST",
        "payload": {"query": "machine learning algorithms", "k": 4},
        "description": "Vector store retrieval test"
    },
    "web_search": {
        "url": "/test/web-search", 
        "method": "POST",
        "payload": {"query": "latest AI research 2025"},
        "description": "Web search functionality test"
    },
    "text_to_speech": {
        "url": "/test/text-to-speech",
        "method": "POST", 
        "payload": {"text": "This is a test of the text to speech functionality using ElevenLabs API."},
        "description": "Text-to-speech conversion test"
    },
    "summarize": {
        "url": "/test/summarize", 
        "method": "POST",
        "payload": {"file_path": TEST_PDF_FILE},
        "description": "Document summarization test",
        "requires_file": TEST_PDF_FILE
    },
    "visualization": {
        "url": "/test/visualization",
        "method": "POST", 
        "payload": {
            "csv_path": TEST_CSV_FILE,
            "query": "Create a bar chart showing the distribution of data"
        },
        "description": "Data visualization test",
        "requires_file": TEST_CSV_FILE
    }
}


CALLS_PER_TOOL = 3
REQUEST_TIMEOUT = 60  # Increased timeout for heavy operations


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"API Status: {health_data.get('status', 'unknown')}")
            print(f"Tools loaded: {health_data.get('tools_loaded', [])}")
            return True
        else:
            print(f"API health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"Cannot reach API: {e}")
        return False


def check_required_files():
    """Check if required test files exist"""
    missing_files = []
    for tool_name, config in TOOLS.items():
        if "requires_file" in config:
            if not os.path.exists(config["requires_file"]):
                missing_files.append((tool_name, config["requires_file"]))
    
    if missing_files:
        print("\nMissing test files:")
        for tool_name, file_path in missing_files:
            print(f"  {tool_name}: {file_path}")
        print("\nSome benchmarks will be skipped.")
    
    return missing_files


def benchmark_tool(name, config):
    """Benchmark a single tool with multiple calls"""
    print(f"\nBenchmarking {name}...")
    print(f"Description: {config['description']}")
    
    # Check if required file exists
    if "requires_file" in config and not os.path.exists(config["requires_file"]):
        print(f"  Skipping - required file not found: {config['requires_file']}")
        return {
            "skipped": True,
            "reason": f"Required file not found: {config['requires_file']}"
        }
    
    times = []
    successes = 0
    errors = []
    url = API_URL + config["url"]
    
    for call_num in range(1, CALLS_PER_TOOL + 1):
        print(f"  Call {call_num}/{CALLS_PER_TOOL}...", end=" ")
        start = time.time()
        
        try:
            if config["method"] == "POST":
                response = requests.post(url, json=config["payload"], timeout=REQUEST_TIMEOUT)
            else:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
            
            response.raise_for_status()
            duration = time.time() - start
            times.append(duration)
            successes += 1
            print(f"OK ({duration:.2f}s)")
            
        except requests.exceptions.Timeout:
            duration = REQUEST_TIMEOUT
            times.append(duration)
            errors.append(f"Call {call_num}: Timeout after {REQUEST_TIMEOUT}s")
            print("TIMEOUT")
            
        except requests.exceptions.RequestException as e:
            duration = time.time() - start
            times.append(duration)
            errors.append(f"Call {call_num}: {str(e)}")
            print(f"ERROR - {str(e)}")
            
        except Exception as e:
            duration = time.time() - start
            times.append(duration)
            errors.append(f"Call {call_num}: Unexpected error - {str(e)}")
            print(f"ERROR - {str(e)}")


    if times:
        result = {
            "calls": CALLS_PER_TOOL,
            "successes": successes,
            "failures": CALLS_PER_TOOL - successes,
            "success_rate": (successes / CALLS_PER_TOOL) * 100,
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times),
            "errors": errors
        }
    else:
        result = {
            "calls": CALLS_PER_TOOL,
            "successes": 0,
            "failures": CALLS_PER_TOOL,
            "success_rate": 0,
            "avg_time": 0,
            "min_time": 0,
            "max_time": 0,
            "total_time": 0,
            "errors": errors
        }
    
    print(f"  Results: {successes}/{CALLS_PER_TOOL} successful, avg: {result['avg_time']:.2f}s")
    return result


def run_benchmark(selected_tools=None):
    """Run benchmark for selected tools or all tools"""
    print("StudyBuddy Tools Benchmark")
    print("=" * 50)
    
    if not check_api_health():
        print("Cannot proceed - API is not available")
        return None
    
    missing_files = check_required_files()
    
    # Filter tools if specific ones are selected
    if selected_tools:
        tools_to_test = {k: v for k, v in TOOLS.items() if k in selected_tools}
        if not tools_to_test:
            print(f"No valid tools found. Available: {list(TOOLS.keys())}")
            return None
    else:
        tools_to_test = TOOLS
    
    print(f"\nRunning benchmark with {CALLS_PER_TOOL} calls per tool")
    print(f"Testing {len(tools_to_test)} tools: {list(tools_to_test.keys())}")
    
    results = {}
    start_time = time.time()
    
    for name, config in tools_to_test.items():
        results[name] = benchmark_tool(name, config)
    
    total_duration = time.time() - start_time
    
    # Calculate overall statistics
    total_calls = sum(r.get('calls', 0) for r in results.values() if not r.get('skipped'))
    total_successes = sum(r.get('successes', 0) for r in results.values() if not r.get('skipped'))
    overall_success_rate = (total_successes / total_calls * 100) if total_calls > 0 else 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        "timestamp": timestamp,
        "benchmark_duration": total_duration,
        "tools_tested": list(tools_to_test.keys()),
        "calls_per_tool": CALLS_PER_TOOL,
        "overall_stats": {
            "total_calls": total_calls,
            "total_successes": total_successes,
            "overall_success_rate": overall_success_rate,
            "total_duration": total_duration
        },
        "individual_results": results
    }
    
    output_path = os.path.join(BENCHMARK_DIR, f"benchmark_results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'=' * 50}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Total calls: {total_calls}")
    print(f"Total successes: {total_successes}")
    print(f"Overall success rate: {overall_success_rate:.1f}%")
    
    print(f"\nPer-tool results:")
    for tool_name, result in results.items():
        if result.get('skipped'):
            print(f"  {tool_name.ljust(25)} - SKIPPED ({result['reason']})")
        else:
            success_rate = result['success_rate']
            avg_time = result['avg_time']
            print(f"  {tool_name.ljust(25)} - {success_rate:5.1f}% success, {avg_time:6.2f}s avg")
    
    print(f"\nDetailed results saved to: {output_path}")
    return output_path


def run_specific_tools(tool_names):
    """Run benchmark for specific tools"""
    invalid_tools = [t for t in tool_names if t not in TOOLS]
    if invalid_tools:
        print(f"Invalid tool names: {invalid_tools}")
        print(f"Available tools: {list(TOOLS.keys())}")
        return None
    
    return run_benchmark(tool_names)


def print_usage():
    """Print usage information"""
    print("\nStudyBuddy Tools Benchmark")
    print("=" * 30)
    print("\nUsage:")
    print("  python benchmark_tools.py [tool1] [tool2] ...")
    print("\nAvailable tools:")
    for name, config in TOOLS.items():
        requires_file = " (requires file)" if "requires_file" in config else ""
        print(f"  {name.ljust(25)} - {config['description']}{requires_file}")
    print(f"\nDefault: {CALLS_PER_TOOL} calls per tool")
    print(f"Timeout: {REQUEST_TIMEOUT}s per request")
    print("\nExamples:")
    print("  python benchmark_tools.py                    # Test all tools")
    print("  python benchmark_tools.py vector_search      # Test only vector search")
    print("  python benchmark_tools.py web_search tts     # Test web search and TTS")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["help", "-h", "--help"]:
            print_usage()
        else:
            tool_names = sys.argv[1:]
            result = run_specific_tools(tool_names)
            sys.exit(0 if result else 1)
    else:
        run_benchmark()