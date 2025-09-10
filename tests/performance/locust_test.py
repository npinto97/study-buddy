from locust import HttpUser, task, between
import random
import os

# Test data for the 5 core tools
SEARCH_QUERIES = [
    "Knowledge Discovery",
    "recommender systems",
    "text categorization",
    "bayesian classifiers",
    "neural networks",
    "Context-aware Explanations"
]

WEB_QUERIES = [
    "latest AI research 2025",
    "python 3.12 features",
    "quantum computing news",
    "machine learning trends",
    "data science tools",
    "AI breakthrough 2025"
]

TTS_TEXTS = [
    "This is a test of the text to speech functionality.",
    "Machine learning is transforming the world of technology.",
    "Data science helps us understand complex patterns.",
    "Artificial intelligence is the future of computing.",
    "Neural networks can solve complex problems."
]

# Test files paths (adjust these to your actual test files)
TEST_PDF_FILE = "./test_files/sample.pdf"
TEST_CSV_FILE = "./test_files/sample_data.csv"

CSV_QUERIES = [
    "Show me a bar chart of the data",
    "Create a histogram of numerical columns",
    "Generate a scatter plot",
    "Make a line chart showing trends",
    "Create a pie chart of exam scores"
]

class StudyBuddyLoadTest(HttpUser):
    wait_time = between(1, 3)

    @task(4)
    def test_vector_search(self):
        """Test vector store retrieval - high frequency"""
        query = random.choice(SEARCH_QUERIES)
        self.client.post("/test/vector-search", json={"query": query, "k": 4})

    @task(4)
    def test_web_search(self):
        """Test web search functionality - high frequency"""
        query = random.choice(WEB_QUERIES)
        self.client.post("/test/web-search", json={"query": query})

    @task(2)
    def test_text_to_speech(self):
        """Test text-to-speech - medium frequency"""
        text = random.choice(TTS_TEXTS)
        self.client.post("/test/text-to-speech", json={"text": text})

    @task(1)
    def test_summarize(self):
        """Test document summarization - low frequency"""
        if os.path.exists(TEST_PDF_FILE):
            self.client.post("/test/summarize", json={"file_path": TEST_PDF_FILE})

    @task(1)
    def test_visualization(self):
        """Test data visualization - low frequency"""
        if os.path.exists(TEST_CSV_FILE):
            query = random.choice(CSV_QUERIES)
            self.client.post("/test/visualization", json={
                "csv_path": TEST_CSV_FILE,
                "query": query
            })

    @task(1)
    def test_health_check(self):
        """Test health endpoint"""
        self.client.get("/health")

    @task(1)
    def test_metrics(self):
        """Test metrics endpoint"""
        self.client.get("/metrics")

class LightLoad(StudyBuddyLoadTest):
    """Light load scenario - more wait time between requests"""
    wait_time = between(3, 6)

class MediumLoad(StudyBuddyLoadTest):
    """Medium load scenario - normal wait time"""
    wait_time = between(1, 3)

class HeavyLoad(StudyBuddyLoadTest):
    """Heavy load scenario - less wait time"""
    wait_time = between(0.5, 1.5)

# if __name__ == "__main__":
#     print("StudyBuddy Load Test")
#     print("Available test classes: StudyBuddyLoadTest, LightLoad, MediumLoad, HeavyLoad")
#     print("Run with: locust -f locust_test.py --host=http://127.0.0.1:8000")
#     print("Example: locust -f locust_test.py --host=http://127.0.0.1:8000 --class-picker LightLoad")