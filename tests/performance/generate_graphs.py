import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import glob
from datetime import datetime
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

BENCHMARK_DIR = "./benchmark_reports"
LOAD_TEST_DIR = "./load_test_reports"
GRAPHS_DIR = "./test_graphs"

os.makedirs(GRAPHS_DIR, exist_ok=True)

def plot_benchmark_results():
    """Generate graphs from benchmark JSON results"""
    print("Looking for benchmark results...")
    
    # Find the latest benchmark result
    json_files = glob.glob(os.path.join(BENCHMARK_DIR, "benchmark_results_*.json"))
    if not json_files:
        print("No benchmark results found in", BENCHMARK_DIR)
        return
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"Using latest benchmark: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    results = data['individual_results']
    timestamp = data['timestamp']
    
    # Extract data for plotting
    tools = []
    success_rates = []
    avg_times = []
    min_times = []
    max_times = []
    
    for tool_name, result in results.items():
        if not result.get('skipped', False):
            tools.append(tool_name.replace('_', ' ').title())
            success_rates.append(result['success_rate'])
            avg_times.append(result['avg_time'])
            min_times.append(result['min_time'])
            max_times.append(result['max_time'])
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Benchmark Results - {timestamp}', fontsize=16, fontweight='bold')
    
    # 1. Success Rate Bar Chart
    bars1 = ax1.bar(tools, success_rates, color='lightgreen', alpha=0.7)
    ax1.set_title('Success Rate by Tool', fontweight='bold')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # 2. Average Response Time Bar Chart
    bars2 = ax2.bar(tools, avg_times, color='lightblue', alpha=0.7)
    ax2.set_title('Average Response Time by Tool', fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, time in zip(bars2, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time:.2f}s', ha='center', va='bottom')
    
    # 3. Response Time Range (Min/Max)
    x_pos = range(len(tools))
    ax3.bar(x_pos, max_times, color='lightcoral', alpha=0.7, label='Max Time')
    ax3.bar(x_pos, min_times, color='darkgreen', alpha=0.8, label='Min Time')
    ax3.set_title('Response Time Range by Tool', fontweight='bold')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(tools, rotation=45)
    ax3.legend()
    
    # 4. Overall Statistics Pie Chart
    total_calls = data['overall_stats']['total_calls']
    total_successes = data['overall_stats']['total_successes']
    total_failures = total_calls - total_successes
    
    if total_calls > 0:
        sizes = [total_successes, total_failures]
        labels = [f'Success\n({total_successes})', f'Failed\n({total_failures})']
        colors = ['lightgreen', 'lightcoral']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Overall Success/Failure Rate', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(GRAPHS_DIR, f'benchmark_results_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Benchmark graph saved: {output_file}")
    plt.show()

def plot_load_test_results():
    """Generate graphs from Locust CSV results"""
    print("Looking for load test results...")
    
    # Find all stats CSV files
    stats_files = glob.glob(os.path.join(LOAD_TEST_DIR, "*_stats.csv"))
    if not stats_files:
        print("No load test results found in", LOAD_TEST_DIR)
        return
    
    # Group files by scenario
    scenarios = {}
    for file in stats_files:
        filename = os.path.basename(file)
        # Extract scenario name from filename (format: ClassName_scenario_timestamp_stats.csv)
        parts = filename.replace('_stats.csv', '').split('_')
        if len(parts) >= 2:
            scenario_name = parts[1]  # Get scenario name
            scenarios[scenario_name] = file
    
    if not scenarios:
        print("Could not parse scenario names from files")
        return
    
    print(f"Found {len(scenarios)} scenarios: {list(scenarios.keys())}")
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Load Test Results Comparison', fontsize=16, fontweight='bold')
    
    scenario_names = []
    avg_response_times = []
    failure_rates = []
    request_rates = []
    
    for scenario_name, file_path in scenarios.items():
        try:
            df = pd.read_csv(file_path)
            
            # Filter out the "Aggregated" row if it exists
            df = df[df['Name'] != 'Aggregated']
            
            scenario_names.append(scenario_name.title())
            
            # Calculate metrics
            avg_response = df['Average Response Time'].mean() if 'Average Response Time' in df.columns else 0
            total_requests = df['Request Count'].sum() if 'Request Count' in df.columns else 0
            total_failures = df['Failure Count'].sum() if 'Failure Count' in df.columns else 0
            failure_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0
            
            avg_response_times.append(avg_response)
            failure_rates.append(failure_rate)
            request_rates.append(total_requests)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not scenario_names:
        print("No valid data found in CSV files")
        return
    
    # 1. Average Response Time by Scenario
    bars1 = ax1.bar(scenario_names, avg_response_times, color='lightblue', alpha=0.7)
    ax1.set_title('Average Response Time by Scenario', fontweight='bold')
    ax1.set_ylabel('Response Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, time in zip(bars1, avg_response_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{time:.0f}ms', ha='center', va='bottom')
    
    # 2. Failure Rate by Scenario
    bars2 = ax2.bar(scenario_names, failure_rates, color='lightcoral', alpha=0.7)
    ax2.set_title('Failure Rate by Scenario', fontweight='bold')
    ax2.set_ylabel('Failure Rate (%)')
    ax2.set_ylim(0, max(failure_rates) * 1.2 if failure_rates else 10)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, rate in zip(bars2, failure_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # 3. Total Requests by Scenario
    bars3 = ax3.bar(scenario_names, request_rates, color='lightgreen', alpha=0.7)
    ax3.set_title('Total Requests by Scenario', fontweight='bold')
    ax3.set_ylabel('Number of Requests')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars3, request_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(request_rates) * 0.01,
                f'{count}', ha='center', va='bottom')
    
    # 4. Performance Trend (if we have data over time)
    # For now, show a simple comparison chart
    x_pos = range(len(scenario_names))
    width = 0.35
    
    ax4.bar([x - width/2 for x in x_pos], avg_response_times, width, 
           label='Avg Response (ms)', alpha=0.7, color='lightblue')
    ax4_twin = ax4.twinx()
    ax4_twin.bar([x + width/2 for x in x_pos], failure_rates, width,
                label='Failure Rate (%)', alpha=0.7, color='lightcoral')
    
    ax4.set_xlabel('Scenarios')
    ax4.set_ylabel('Response Time (ms)', color='blue')
    ax4_twin.set_ylabel('Failure Rate (%)', color='red')
    ax4.set_title('Response Time vs Failure Rate', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(scenario_names, rotation=45)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(GRAPHS_DIR, f'load_test_comparison_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Load test graph saved: {output_file}")
    plt.show()

def plot_detailed_load_test(csv_file):
    """Generate detailed graphs for a specific load test result"""
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
    
    try:
        df = pd.read_csv(csv_file)
        filename = os.path.basename(csv_file).replace('_stats.csv', '')
        
        # Filter out aggregated row
        df = df[df['Name'] != 'Aggregated']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Detailed Load Test Results - {filename}', fontsize=16, fontweight='bold')
        
        # 1. Requests per endpoint
        if 'Request Count' in df.columns:
            ax1.bar(df['Name'], df['Request Count'], color='lightgreen', alpha=0.7)
            ax1.set_title('Requests per Endpoint', fontweight='bold')
            ax1.set_ylabel('Number of Requests')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Response times per endpoint
        if 'Average Response Time' in df.columns:
            ax2.bar(df['Name'], df['Average Response Time'], color='lightblue', alpha=0.7)
            ax2.set_title('Average Response Time per Endpoint', fontweight='bold')
            ax2.set_ylabel('Response Time (ms)')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Failure count
        if 'Failure Count' in df.columns:
            ax3.bar(df['Name'], df['Failure Count'], color='lightcoral', alpha=0.7)
            ax3.set_title('Failures per Endpoint', fontweight='bold')
            ax3.set_ylabel('Number of Failures')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Response time percentiles
        percentile_cols = [col for col in df.columns if '% Response Time' in col]
        if percentile_cols:
            for i, col in enumerate(percentile_cols):
                ax4.plot(df['Name'], df[col], marker='o', label=col, alpha=0.8)
            ax4.set_title('Response Time Percentiles', fontweight='bold')
            ax4.set_ylabel('Response Time (ms)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend()
        
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(GRAPHS_DIR, f'detailed_{filename}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Detailed graph saved: {output_file}")
        plt.show()
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

def main():
    """Main function to generate all graphs"""
    print("StudyBuddy Test Results Visualizer")
    print("=" * 50)
    
    # Create graphs directory
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    # Generate benchmark graphs
    try:
        plot_benchmark_results()
    except Exception as e:
        print(f"Error generating benchmark graphs: {e}")
    
    # print()
    
    # Generate load test comparison graphs
    try:
        plot_load_test_results()
    except Exception as e:
        print(f"Error generating load test graphs: {e}")
    
    print(f"\nAll graphs saved to: {GRAPHS_DIR}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark":
            plot_benchmark_results()
        elif sys.argv[1] == "loadtest":
            plot_load_test_results()
        elif sys.argv[1].endswith('.csv'):
            plot_detailed_load_test(sys.argv[1])
        else:
            print("Usage:")
            print("  python visualize_results.py                    # Generate all graphs")
            print("  python visualize_results.py benchmark          # Only benchmark graphs")
            print("  python visualize_results.py loadtest           # Only load test graphs")
            print("  python visualize_results.py path/to/stats.csv  # Detailed graph for specific CSV")
    else:
        main()