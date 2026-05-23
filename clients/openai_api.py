import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

def get_client(model_name,office):
    os.environ["MODEL_NAME"] = model_name
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    return client

def openai_chat(prompt,model_name="",office=False,think=False):
    client = get_client(model_name,office)

    print(f"[openai_chat] prompt:{prompt}")

    completion = client.chat.completions.create(
        timeout=120,
        model=os.environ["MODEL_NAME"],
        response_format={"type": "json_object"},
        # extra_body={"enable_thinking": think},
        temperature=0.01,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    try:
        print(f"[openai_chat] completion:{completion}")
        answer = completion.choices[0].message.content

        return answer
    except Exception as e:
        print(f"error:{e},{completion}")
        return ""


if __name__ == '__main__':
    prompt = "1+1=?json"
    ans = openai_chat(prompt=prompt, model_name='deepseek-v4-flash',think=False)
    print(ans)



