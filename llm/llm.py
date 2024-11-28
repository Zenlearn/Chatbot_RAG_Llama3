from config.config_aws_bedrock import BedrockClient as client
from config.config_env import MODEL_ID
import json
# from config.config_ollama import llama

def llama3(prompt: str) -> str:
    try:
        #* LLAMA 3 8B model [OllamaLLM]
        # return llama.invoke(prompt)

        #* LLAMA 3 model [AWS Bedrock]
        request_body = {"prompt": prompt}

        # Make a request to Bedrock for text generation
        response = client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body),
        )

        # Parse the response
        response_body = response["body"].read().decode("utf-8")

        # Parse the JSON response
        response_json = json.loads(response_body)
        # dict_keys(['generation', 'prompt_token_count', 'generation_token_count', 'stop_reason'])
        return response_json.get("generation", "No text generated.")
    
    
    except Exception as e:
        print(f"Error querying Llama 3 model: {e}")
        return None

def prompt(query: str, vectors: list[str]) -> str:
    vectors_str = "\n".join(vectors)
    return llama3(f"Query: '{query}'\n\nReference Vectors:\n{vectors_str}\n\nPlease generate the output based question and reference vectors above, and provide the answer only.")