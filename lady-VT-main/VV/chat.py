import openai


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = "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"


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.9, tokens=400, freq_pen=2.0, pres_pen=2.0, stop=['BRITAIN:', 'CHAT:']):
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temp,
        max_tokens=tokens,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stop=stop)
    text = response['choices'][0]['text'].strip()
    return text






