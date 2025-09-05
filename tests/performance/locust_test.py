from locust import HttpUser, task, between
import random
import os

SEARCH_QUERIES = [
    "machine learning algorithms",
    "natural language processing", 
    "computer vision techniques",
    "data science methods",
    "artificial intelligence research",
    "deep learning frameworks",
    "statistical analysis",
    "python programming",
    "database optimization",
    "cloud computing architecture"
]

WEB_SEARCH_QUERIES = [
    "latest AI research 2024",
    "machine learning trends",
    "ChatGPT updates",
    "Python 3.12 features", 
    "Docker best practices",
    "JavaScript frameworks 2024",
    "cybersecurity threats",
    "quantum computing news",
    "blockchain applications",
    "renewable energy technology"
]

CODE_SAMPLES = [
    "print('Hello World')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')",
    "import pandas as pd\ndf = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})\nprint(df.head())",
    "import numpy as np\narr = np.array([1, 2, 3, 4, 5])\nprint(f'Mean: {arr.mean()}')",
    "from datetime import datetime\nnow = datetime.now()\nprint(f'Current time: {now}')",
    "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3], [1, 4, 9])\nplt.title('Simple Plot')"
]

class StudyBuddyLoadTest(HttpUser):
    """Default load test scenario"""
    wait_time = between(1, 3)

    @task(30)
    def test_vector_search(self):
        query = random.choice(SEARCH_QUERIES)
        self.client.post("/test/vector-search", json={"query": query, "k": random.randint(3, 6)})

    @task(20)
    def test_web_search(self):
        query = random.choice(WEB_SEARCH_QUERIES)
        self.client.post("/test/web-search", json={"query": query})

    @task(15)
    def test_code_execution(self):
        code = random.choice(CODE_SAMPLES)
        self.client.post("/test/execute-code", json={"code": code})

    @task(10)
    def test_document_processing(self):
        test_files = [
            "./test_files/sample.pdf",
            "./test_files/sample.txt", 
            "./test_files/sample.docx"
        ]
        available_files = [f for f in test_files if os.path.exists(f)]
        if available_files:
            file_path = random.choice(available_files)
            self.client.post("/test/extract-text", json={"file_path": file_path})

    @task(5)
    def test_data_visualization(self):
        csv_files = [
            "./test_files/sample_data.csv",
            "./test_files/sales_data.csv",
            "./test_files/analytics.csv"
        ]
        viz_queries = [
            "Create a bar chart of the top 5 categories",
            "Show a time series plot of the data",
            "Generate a correlation heatmap",
            "Create a pie chart of the distribution",
            "Plot a scatter plot with trend line"
        ]
        available_csvs = [f for f in csv_files if os.path.exists(f)]
        if available_csvs:
            csv_path = random.choice(available_csvs)
            query = random.choice(viz_queries)
            self.client.post("/test/visualization", json={"csv_path": csv_path, "query": query})

    @task(10)
    def check_health_and_metrics(self):
        self.client.get("/health")
        self.client.get("/metrics")

    @task(5)
    def test_batch_operations(self):
        self.client.post("/test/batch")


# Different scenarios for flexibility
class LightLoad(StudyBuddyLoadTest):
    """Simulates low traffic"""
    wait_time = between(2, 5)


class HeavyLoad(StudyBuddyLoadTest):
    """Simulates high traffic"""
    wait_time = between(0.5, 2)


class StressTest(StudyBuddyLoadTest):
    """Pushes the system to its limits"""
    wait_time = between(0.1, 1)

    @task(50)
    def stress_vector_search(self):
        self.test_vector_search()

    @task(30)
    def stress_web_search(self):
        self.test_web_search()

    @task(20)
    def stress_code_execution(self):
        self.test_code_execution()


if __name__ == "__main__":
    print("Load testing configuration loaded")
    print("Run with: locust -f locust_test.py --host=http://localhost:8000")
    print("Available test classes:")
    print("  - StudyBuddyLoadTest (default)")
    print("  - LightLoad")
    print("  - HeavyLoad")
    print("  - StressTest")
