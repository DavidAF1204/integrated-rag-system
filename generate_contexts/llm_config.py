from openai import AzureOpenAI

llm = AzureOpenAI(
    api_key="AZURE_API_KEY",
    api_version="AZURE_API_VERSION",
    azure_endpoint="AZURE_ENDPOINT"
)
MODEL_NAME="AZURE_MODEL_NAME"
TEMPERATURE=0.0