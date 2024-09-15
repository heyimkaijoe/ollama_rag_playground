import ollama
import os
import json
from numpy import linalg, dot

def parse_paragraph(filename):
    with open(filename) as file:
        return [ line.strip() for line in file if line.strip() ]
    
def calc_embeddings(paragraphs):
    return [ ollama.embeddings(model='mistral', prompt=data)['embedding'] for data in paragraphs ]

def cache_embeddings(filename, paragraphs):
    embedded_file = f'cache/{filename}.json'

    if os.path.isfile(embedded_file):
        with open(embedded_file) as file:
            return json.load(file)
    
    os.makedirs('cache', exist_ok=True)

    embeddings = calc_embeddings(paragraphs)

    with open(embedded_file, 'w') as file:
        json.dump(embeddings, file)

    return embeddings

def calc_similar_vectors(v, vectors):
    v_norm = linalg.norm(v)
    scores = [ dot(v,item) / ((v_norm) * linalg.norm(item)) for item in vectors ]
    return sorted(enumerate(scores), reverse=True, key=lambda x: x[1])
    
if __name__ == '__main__':
    doc = 'maxwell_mead.txt'
    paragraphs = parse_paragraph(doc)
    embeddings = cache_embeddings(doc, paragraphs)

    prompt = input('What do you want to ask?\n>>> ')

    while prompt.lower() != 'bye':
        prompt_embedding = ollama.embeddings(model='mistral', prompt=prompt)['embedding']
        similar_vectors = calc_similar_vectors(prompt_embedding, embeddings)[:3]
        system_prompt = (
            "If you don't know the answer, please feel free to say 'I don't know'. Here's the context: " +
            '\n'.join(paragraphs[vector[0]] for vector in similar_vectors)
        )

        response = ollama.chat(
            model='mistral',
            messages=[
                { 'role': 'system', 'content': system_prompt },
                { 'role': 'user', 'content': prompt },
            ],
        )

        print(response['message']['content'])
        prompt = input('>>> ')
