import json
from openai import OpenAI
from configs import Config

config = Config()

client = OpenAI(api_key=config.api_key, base_url=config.base_url)

def generate_health_plan(user_data: dict) -> dict:

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": config.prompt},
                {"role": "user", "content": json.dumps(user_data)},
            ],
            stream=False,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            response_format={'type': 'json_object'}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        raise RuntimeError(f"API failed: {str(e)}")