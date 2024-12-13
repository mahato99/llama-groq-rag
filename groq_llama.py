from groq import Groq
import os 
from dotenv import load_dotenv
load_dotenv()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models in today's landscape",
        }
    ],
    model="llama3-8b-8192",
    stream=False,
)

print(chat_completion.choices[0].message.content)