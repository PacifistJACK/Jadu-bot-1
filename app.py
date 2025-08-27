from flask import Flask, request, render_template, jsonify
import json
from groq import Groq

app = Flask(__name__)

# Initialize the Groq client with your API key
client = Groq(api_key="")

# Load the data from the JSON file
def load_data(file_path='data.json'):
    with open(file_path, 'r') as file:
        return json.load(file)

# Get a response from the Llama model using Groq API
def get_llama_response(prompt):
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
            top_p=1.0,
            stream=True,
            stop=None
        )

        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        return response.strip()
    except Exception as e:
        return f"Error: {e}"

def get_direct_response(question, data):
    for entry in data:
        if entry["instruction"].lower() == question.lower():
            return entry["output"]
    return None

def get_best_match(question, data):
    prompt = "Based on the provided information, respond to the following question and respnse should be pure Answer no need to metion about data provided\n"
    for entry in data:
        prompt += f"Q: {entry['instruction']} A: {entry['output']}\n"

    prompt += f"\nUser's Question: {question}\nAnswer:"
    return get_llama_response(prompt)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    data = load_data()

    response = get_direct_response(user_input, data)
    if not response:
        response = get_best_match(user_input, data)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
