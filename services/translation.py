from langdetect import detect
from config.config_env import BHASHINI_API_KEY, BHASHINI_API_URL
import requests

# * Bhashini API Integration
{  # api_url = 'https://demo-api.models.ai4bharat.org/inference/translation/v2'
    #     payload = {
    #         "controlConfig": {"dataTracking": True},
    #         "input": [{"source": message_body}],
    #         "config": {
    #             "language": {
    #                 "sourceLanguage": source_language,
    #                 "targetLanguage": target_language,
    #             },
    #         },
    #     }
    #     headers = {
    #         'accept': '*/*',
    #         'content-type': 'application/json',
    #         'origin': 'https://models.ai4bharat.org',
    #         'referer': 'https://models.ai4bharat.org/',
    #         'ulcaApiKey': ai4bharat_api_key
    #     }
}


def detect_language(text: str) -> str:
    """
    Detect the language of a given text.
    """
    return detect(text)


def translate(input_data: str) -> str:
    """
    Translate content to English using Bhashini API.
    """
    source_language = detect_language(input_data)
    if source_language == "en":
        return input_data

    target_language = "en"
    payload = {
        "controlConfig": {"dataTracking": True},
        "input": [{"source": input_data}],
        "config": {
            "language": {
                "sourceLanguage": source_language,
                "targetLanguage": target_language,
            },
        },
    }

    headers = {
        "accept": "*/*",
        "content-type": "application/json",
        "origin": "https://models.ai4bharat.org",
        "referer": "https://models.ai4bharat.org/",
        "ulcaApiKey": BHASHINI_API_KEY,
    }

    response = requests.post(BHASHINI_API_URL, json=payload, headers=headers)
    response_data = response.json()

    if response.status_code == 200 and "output" in response_data:
        return response_data["output"][0]["target"]
    else:
        raise Exception(
            f"Translation API request failed with status code {response.status_code}: {response_data}"
        )
