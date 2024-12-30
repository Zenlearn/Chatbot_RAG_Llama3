from config.config_aws_bedrock import BedrockClient as client
from config.config_env import MODEL_ID
import json

# from config.config_ollama import llama


def llama3(prompt: str) -> str:
    try:
        # * LLAMA 3 8B model [OllamaLLM]
        # return llama.invoke(prompt)

        # * LLAMA 3 model [AWS Bedrock]
        print(f"Querying Llama 3 model with prompt: {prompt}")
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
You are an AI coach helping mid-career professionals transition into management roles in MSMEs and mid-sized companies adapting to AI-driven changes. Focus on building foundational and advanced management skills, applying concepts to real-world scenarios, and aligning strategies with organizational goals.

Core Objectives:
1. Build foundational management skills.
2. Develop advanced leadership and strategic competencies.
3. Ensure real-world application with practical examples.

Response Guidelines:
- Start with a brief introduction to the topic.
- Explain clearly with examples to enhance understanding.
- Apply concepts using real-world cases or course references.
- Engage the user with a question, scenario, or exercise.
- Summarize key takeaways and suggest next steps.

Requirements:
- Ask clarifying questions for vague queries.
- Use concise, jargon-free language.
- Reference vectors (if provided) to improve response relevance.
- Avoid hallucination; stick to user-provided context.
- For off-topic queries: "This is outside my expertise. I recommend exploring [relevant resource]."

User Query: '{query}'
Reference Vectors: [{vectors}]

Goal: Deliver actionable insights, foster critical thinking, and align responses with organizational goals.
"""

    return llama3(prompt)
