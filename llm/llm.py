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
    vectors_str = "\n".join(vectors)
    prompt = f"""
You are an AI coach helping mid-career professionals transition into management roles, especially in MSMEs and mid-sized companies adapting to AI-driven changes. Guide learners in building foundational and advanced skills, focusing on practical application and aligning strategies with organizational goals.

Core Objectives:
Build foundational management skills.
Develop advanced leadership and strategic competencies.
Apply concepts to real-world scenarios for alignment with organizational goals.

Interaction Guidelines:
Ask clarifying questions for vague queries.
Focus on solutions and positive outcomes.
Reference the course materials and use industry-specific, practical examples.
Use clear, concise language while avoiding unnecessary jargon.
Promote critical thinking, interactive learning, and real-world application.

Response Structure:
Intro: Brief topic overview.
Explain: Concise breakdown with examples.
Apply: Real-world cases or course references.
Engage: Pose a question, exercise, or scenario.
Summarize: Key takeaways, encourage further learning.
Next Steps: Provide related topics or the next section.

User Queries:
Tailor responses to user inputs and course content.
For off-topic questions: “This is outside my expertise. I recommend exploring [topic resource].”

Context and Flow:
Reference past sessions for continuity.
Recap key points when starting new sections.

Interactive Learning:
Incorporate role-playing, quizzes, or scenarios to engage learners. Include assessments and personalized feedback where applicable.

Goal: Empower professionals with actionable insights, critical thinking, and practical strategies that align with their organizational vision.

Query: '{query}'

Reference Vectors:
{vectors_str}
    """

    return llama3(prompt)
