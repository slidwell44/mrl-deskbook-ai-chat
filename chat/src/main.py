import openai
from openai.types.responses.response import Response

from config import settings

client = openai.OpenAI(api_key=settings.openai.API_KEY)

response: Response = client.responses.create(
    model="gpt-5-nano",
    input="write a haiku about ai",
    store=True,
)

print(response.output_text)
