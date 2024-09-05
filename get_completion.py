from openai import OpenAI
import openai
import anthropic
import replicate
import time



def get_completion_ollama(prompt,system_content, model="unsloth_model:latest", client = openai.OpenAI(base_url='http://localhost:11434/v1') ):
    system_prompt = {
        "role": "system", 
        "content": system_content 
    }
    user_message = {"role": "user", "content": prompt}
    messages = [system_prompt, user_message]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content





def get_completion_llama_replicate(prompt, system_message, model = "meta/meta-llama-3-70b-instruct"):
    input = {
        "top_p": 1,
        "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|> <|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "min_tokens": 0,
        "temperature": 0,
        "prompt_template": "{prompt}",
        "presence_penalty": 0,
        "frequency_penalty":0
    }
    
    prediction = replicate.models.predictions.create(model,input=input)

    for i in range(20):
        prediction.reload()
        if prediction.status in {"failed", "canceled"}:
            raise ValueError("prediction failed or cancelled")
            break
            
        elif prediction.status in {"succeeded"}:
            output = "".join(prediction.output)
            break
        time.sleep(1)
    
    return(output)
    
    

def get_completion_gpt(prompt, system_content, model="gpt-4", client = openai.OpenAI()):
    system_prompt = {
        "role": "system", 
        "content": system_content
    }
    user_message = {"role": "user", "content": prompt}
    messages = [system_prompt, user_message]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def get_completion_claude3(prompt, system_content, model = 'claude-3-opus-20240229', client = anthropic.Anthropic()):
    system_prompt = system_content
    message = client.messages.create(
    model=model,
    max_tokens=1000,
    temperature=0.0,
    system=system_prompt,
    messages=[
        {"role": "user", "content": prompt}
    ]
    )
    return message.content[0].text
