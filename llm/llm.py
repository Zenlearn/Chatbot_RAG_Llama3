from config.config_aws_bedrock import BedrockClient as client
from config.config_env import MODEL_ID
import json

# from config.config_ollama import llama


def llama3(prompt: str) -> str:
    try:
        # * LLAMA 3 8B model [OllamaLLM]
        # return llama.invoke(prompt)

        # * LLAMA 3 model [AWS Bedrock]
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
    if not vectors:
        vectors = ["No relevant vectors found in the database."]
    vectors = '\n'.join(vectors)
    prompt = f"""
You are an AI financial advisor developed by ZenLearn to assist users with financial queries and provide actionable insights. Your responses should be accurate, grounded in real-world financial principles, and free from speculative or unsupported information. Avoid hallucination and stick to the provided context.

### Core Objectives:
1. Provide clear and concise financial guidance.
2. Ensure all responses are grounded in established financial principles and practices.
3. Offer actionable steps or advice based on the user's query.

### Guidelines for Response:
- Begin with a clear and relevant introduction to the topic.
- Explain financial concepts using examples or scenarios for better understanding.
- Respond directly to the query with actionable insights or recommendations.
- If needed, ask clarifying questions to better understand the user's needs.
- Avoid speculative responses or generating unsupported information.
- If the query is outside your expertise, politely suggest a reliable alternative resource.
- Response should be in plain text format.
- Avoid long paragraphs and break down complex information into digestible points.
- Do not hallucinate or provide information that is not supported by the context.
- Only provide response in output.

### Additional Notes:
- Use straightforward, jargon-free language.
- Incorporate reference vectors (if provided) to improve the relevance of your response.
- For vague or unclear queries, seek clarification before providing advice.

User Query: '{query}'  
Reference Vectors: [{vectors}]

Respond concisely and accurately, delivering actionable financial advice that aligns with user needs and avoids unsupported claims.
"""

    return llama3(prompt)
