import os
from dotenv import load_dotenv
from google.ai import generativelanguage as glm
from google.api_core.client_options import ClientOptions

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
# Create a client
client = glm.ModelServiceClient(
    client_options=ClientOptions(api_key=api_key)
)

print("Listing available models:")
request = glm.ListModelsRequest()
page_result = client.list_models(request=request)
for response in page_result:
    print(response.name)
