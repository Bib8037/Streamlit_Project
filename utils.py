import openai

def get_initial_message():
    messages=[
            {"role": "system", "content": "You are an Assistant who is a highly intelligent chatbot designed to help users answer process engineering technical questions."},
            {"role": "user", "content": "I want to learn AI"},
            {"role": "assistant", "content": "Thats awesome, what do you want to know aboout AI"}
        ]
    return messages

def get_chatgpt_response(messages, engine='gpt-35'):
    #print("model: ", engine)
    response = openai.ChatCompletion.create(
    engine=engine,
    messages=messages,
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    )
    return  response['choices'][0]['message']['content']



def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages