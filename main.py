"""Simple api to get a response from the GPT-3.5-turbo model"""
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from pydantic import BaseModel


app = FastAPI()
client = OpenAI()

with open("system_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

class Chat(BaseModel):
    """CHAT MODEL"""
    type: str
    content: str


class Request(BaseModel):
    """REQUEST MODEL"""
    chats: list[Chat]


@app.post("/", response_class=PlainTextResponse)
def get_response(request: Request):
    """Get a response from the GPT-3.5-turbo model"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            *[{"role": chat.type, "content": chat.content} for chat in request.chats]
        ],
        temperature=1.25,
        max_tokens=512,
        top_p=0.8,
        frequency_penalty=0.2,
        presence_penalty=0.75,
    )
    return response.choices[0].message.content
