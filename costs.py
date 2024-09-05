from openai import OpenAI
import openai
import anthropic
import replicate
import time

def get_completion_gpt_with_cost(prompt, system_content, model="gpt-4", client=openai.OpenAI(), encoding_name = "cl100k_base"):
    """
    This one returns both the completions and the cost.
    """
    system_prompt = {"role": "system", "content": system_content}
    user_message = {"role": "user", "content": prompt}
    messages = [system_prompt, user_message]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    # pricings (https://openai.com/api/pricing/)
    if model == 'gpt-4':
        input_cost_per_1M = 30
        output_cost_per_1M = 60
    elif model == 'gpt-4-turbo':
        input_cost_per_1M = 10
        output_cost_per_1M = 30
    elif model == 'gpt-4o':
        input_cost_per_1M = 5
        output_cost_per_1M = 15
    elif model == 'gpt-3.5-turbo-0125':
        input_cost_per_1M = 0.5
        output_cost_per_1M = 1.5
    elif model == 'gpt-3.5-turbo-instruct':
        input_cost_per_1M = 1.5
        output_cost_per_1M = 2
    else:
        raise ValueError("Unsupported model")
    

    input_cost = (input_tokens / 1000000) * input_cost_per_1M
    output_cost = (output_tokens / 1000000) * output_cost_per_1M
    total_cost = input_cost + output_cost

    #print(f"Input tokens: {input_tokens}")
    #print(f"Output tokens: {output_tokens}")
    #print(f"Total cost: ${total_cost:.6f}")

    return response.choices[0].message.content, input_tokens, output_tokens, total_cost

##########################################################################

def get_completion_claude3_with_cost(prompt, system_content, model = 'claude-3-opus-20240229', client = anthropic.Anthropic()):
    """
    Returns both completion and cost.
    """
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

    # pricings https://www.anthropic.com/api
    if model == 'claude-3-opus-20240229':
        input_cost_per_1M = 15
        output_cost_per_1M = 75
    elif model == 'claude-3-sonnet-20240229':
        input_cost_per_1M = 3
        output_cost_per_1M = 15
    elif model == 'claude-3-haiku-20240307':
        input_cost_per_1M = 0.25
        output_cost_per_1M = 1.25
    elif model == 'claude-3-5-sonnet-20240620':
        input_cost_per_1M = 3
        output_cost_per_1M = 15
    else:
        raise ValueError("Unsupported model")

    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens

    input_cost = (input_tokens / 1000000) * input_cost_per_1M
    output_cost = (output_tokens / 1000000) * output_cost_per_1M
    total_cost = input_cost + output_cost
    #print(f"Total cost: ${total_cost:.6f}")

    return message.content[0].text, input_tokens, output_tokens, total_cost

##########################################################################

def get_completion_llama_replicate_with_cost(prompt, system_message, model = "meta/meta-llama-3-70b-instruct"):
    input = {
        "top_p": 1,
        "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "min_tokens": 0,
        "temperature": 0,
        "prompt_template": "{prompt}",
        "presence_penalty": 0,
        "frequency_penalty":0
    }
    
    if model == "meta/meta-llama-3-70b-instruct":
        input_cost_per_1M = 0.65
        output_cost_per_1M = 2.75
    elif model == "meta/meta-llama-3-8b-instruct":
        input_cost_per_1M = 0.05
        output_cost_per_1M = 0.25
    else:
        raise ValueError("Unsupported model")
        

    prediction = replicate.models.predictions.create(model,input=input)
    
    for i in range(20):
        prediction.reload()
        if prediction.status in {"failed", "canceled"}:
            raise ValueError("prediction failed or cancelled")
            break
            
        elif prediction.status in {"succeeded"}:
            output = "".join(prediction.output)
            id_ = prediction.id
            metric = replicate.predictions.get(id_).metrics
            input_tokens = metric['input_token_count']
            output_tokens = metric['output_token_count']

            input_cost = (input_tokens / 1000000) * input_cost_per_1M
            output_cost = (output_tokens / 1000000) * output_cost_per_1M
            total_cost = input_cost + output_cost
            break

        # Wait for 2 seconds and then try again.
        time.sleep(1)
    
    return output, input_tokens, output_tokens, total_cost

