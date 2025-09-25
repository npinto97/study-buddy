import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import glob
from datetime import datetime
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

BENCHMARK_DIR = "./benchmark_reports"
LOAD_TEST_DIR = "./load_test_reports"
GRAPHS_DIR = "./test_graphs"

os.makedirs(GRAPHS_DIR, exist_ok=True)

def save_and_close(fig, filename):
    output_file = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved: {output_file}")
    plt.close(fig)

def plot_benchmark_results():
    json_files = glob.glob(os.path.join(BENCHMARK_DIR, "benchmark_results_*.json"))
    if not json_files:
        print("No benchmark results found")
        return

    latest_file = max(json_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        data = json.load(f)

    results = data['individual_results']
    timestamp = data['timestamp']

    tools, success_rates, avg_times, min_times, max_times = [], [], [], [], []
    for tool_name, result in results.items():
        if not result.get('skipped', False):
            tools.append(tool_name.replace('_', ' ').title())
            success_rates.append(result['success_rate'])
            avg_times.append(result['avg_time'])
            min_times.append(result['min_time'])
            max_times.append(result['max_time'])

    sorted_data = sorted(zip(tools, success_rates, avg_times, min_times, max_times), key=lambda x: x[1])
    tools, success_rates, avg_times, min_times, max_times = zip(*sorted_data)

    # 1. Success Rate
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(tools, success_rates, color='lightgreen', alpha=0.7)
    ax.set_title(f'Success Rate by Tool - {timestamp}', fontweight='bold')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 105)
    ax.tick_params(axis='x', rotation=45)
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+1, f'{rate:.1f}%', ha='center', va='bottom')
    save_and_close(fig, f'benchmark_success_rate_{timestamp}.png')

    # 2. Average Response Time
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(tools, avg_times, color='lightblue', alpha=0.7)
    ax.set_title(f'Average Response Time by Tool - {timestamp}', fontweight='bold')
    ax.set_ylabel('Time (seconds)')
    ax.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.1, f'{t:.2f}s', ha='center', va='bottom')
    save_and_close(fig, f'benchmark_avg_time_{timestamp}.png')

    # 3. Response Time Range
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = range(len(tools))
    ax.bar(x_pos, max_times, color='lightcoral', alpha=0.7, label='Max Time')
    ax.bar(x_pos, min_times, color='darkgreen', alpha=0.8, label='Min Time')
    ax.set_title(f'Response Time Range by Tool - {timestamp}', fontweight='bold')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tools, rotation=45)
    ax.legend()
    save_and_close(fig, f'benchmark_time_range_{timestamp}.png')

    # 4. Overall Success/Failure Pie Chart
    fig, ax = plt.subplots(figsize=(6, 6))
    total_calls = data['overall_stats']['total_calls']
    total_successes = data['overall_stats']['total_successes']
    total_failures = total_calls - total_successes
    if total_calls > 0:
        ax.pie([total_successes, total_failures], labels=[f'Success\n({total_successes})', f'Failed\n({total_failures})'],
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Overall Success/Failure Rate - {timestamp}', fontweight='bold')
    save_and_close(fig, f'benchmark_overall_pie_{timestamp}.png')


def plot_load_test_results():
    stats_files = glob.glob(os.path.join(LOAD_TEST_DIR, "*_stats.csv"))
    if not stats_files:
        print("No load test results found")
        return

    scenarios = {}
    for file in stats_files:
        filename = os.path.basename(file)
        parts = filename.replace('_stats.csv', '').split('_')
        if len(parts) >= 2:
            scenario_name = parts[1]
            scenarios[scenario_name] = file

    scenario_names, avg_response_times, failure_rates, request_rates = [], [], [], []
    for scenario_name, file_path in scenarios.items():
        df = pd.read_csv(file_path)
        # Rimuovi la riga Aggregated e gli endpoint di monitoraggio
        df = df[df['Name'] != 'Aggregated']
        df = df[~df['Name'].str.contains('health|metrics', case=False)]
        
        scenario_names.append(scenario_name.title())
        avg_response_times.append(df['Average Response Time'].mean())
        total_requests = df['Request Count'].sum()
        total_failures = df['Failure Count'].sum()
        failure_rates.append((total_failures / total_requests * 100) if total_requests > 0 else 0)
        request_rates.append(total_requests)

    sorted_data = sorted(zip(scenario_names, avg_response_times, failure_rates, request_rates), key=lambda x: x[1])
    scenario_names, avg_response_times, failure_rates, request_rates = zip(*sorted_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Avg Response Time
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(scenario_names, avg_response_times, color='lightblue', alpha=0.7)
    ax.set_title('Average Response Time by Scenario', fontweight='bold')
    ax.set_ylabel('Response Time (ms)')
    ax.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, avg_response_times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+10, f'{t:.0f}ms', ha='center', va='bottom')
    save_and_close(fig, f'loadtest_avg_time_{timestamp}.png')

    # 2. Failure Rate
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(scenario_names, failure_rates, color='lightcoral', alpha=0.7)
    ax.set_title('Failure Rate by Scenario', fontweight='bold')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_ylim(0, max(failure_rates) * 1.2 if failure_rates else 10)
    ax.tick_params(axis='x', rotation=45)
    for bar, rate in zip(bars, failure_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.1, f'{rate:.1f}%', ha='center', va='bottom')
    save_and_close(fig, f'loadtest_failure_rate_{timestamp}.png')

    # 3. Total Requests
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(scenario_names, request_rates, color='lightgreen', alpha=0.7)
    ax.set_title('Total Requests by Scenario', fontweight='bold')
    ax.set_ylabel('Number of Requests')
    ax.tick_params(axis='x', rotation=45)
    for bar, count in zip(bars, request_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+max(request_rates)*0.01, f'{count}', ha='center', va='bottom')
    save_and_close(fig, f'loadtest_requests_{timestamp}.png')


def plot_failure_rate_per_tool():
    stats_files = glob.glob(os.path.join(LOAD_TEST_DIR, "*_stats.csv"))
    if not stats_files:
        print("No load test results found for per-tool analysis")
        return

    all_data = []

    for file in stats_files:
        try:
            filename = os.path.basename(file)
            parts = filename.replace('_stats.csv', '').split('_')
            scenario_name = parts[1].title()

            df = pd.read_csv(file)
            df = df[df['Name'] != 'Aggregated']

            for _, row in df.iterrows():
                tool_name = os.path.basename(row['Name']).replace('-', ' ').title()
                
                # Salta gli endpoint di monitoraggio
                if 'health' in tool_name.lower() or 'metrics' in tool_name.lower():
                    continue

                total_requests = row['Request Count']
                failed_requests = row['Failure Count']
                failure_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0

                all_data.append({
                    'Scenario': scenario_name,
                    'Tool': tool_name,
                    'Failure Rate': failure_rate
                })
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    if not all_data:
        print("No valid data to plot per-tool failure rates.")
        return

    df_plot = pd.DataFrame(all_data)
    
    scenario_order = ['Light', 'Medium', 'Heavy', 'Stress']
    df_plot['Scenario'] = pd.Categorical(df_plot['Scenario'], categories=scenario_order, ordered=True)
    df_plot = df_plot.sort_values(by=['Tool', 'Scenario']).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        x='Tool',
        y='Failure Rate',
        hue='Scenario',
        data=df_plot,
        ax=ax,
        palette='viridis'
    )

    ax.set_title('Failure Rate per Tool and Scenario', fontweight='bold')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_ylim(0, 105)
    
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')
    
    ax.set_xlabel('Tool')
    ax.legend(title='Scenario', loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_and_close(fig, f'tool_failure_rates_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')


def main():
    plot_benchmark_results()
    plot_load_test_results()
    plot_failure_rate_per_tool()

if __name__ == "__main__":
    main()