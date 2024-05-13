from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__, static_url_path='/static')

# Load translation models and tokenizers for Odia to Hindi and English to Odia
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load models and tokenizers for Odia to Hindi translation
odia_to_hindi_model = load_model('model.pkl')
odia_to_hindi_tokenizer = load_tokenizer('tokenizer.pkl')

# Load models and tokenizers for English to Odia translation
english_to_odia_model = load_model('Eng_odia_model.pkl')
english_to_odia_tokenizer = load_tokenizer('tokenizer_eng_odia.pkl')

# Translation function for Odia to Hindi
def odia_to_hindi_translate(source_text):
    input_ids = odia_to_hindi_tokenizer.encode(source_text, return_tensors="pt")
    translated_ids = odia_to_hindi_model.generate(input_ids)
    translated_text = odia_to_hindi_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    #translated_text = tokenizer.translate(source_text)
    return translated_text

# Translation function for English to Odia
def english_to_odia_translate(source_text):
    input_ids_e_o = english_to_odia_tokenizer.encode(source_text, return_tensors="pt")
    translated_ids_e_o = english_to_odia_model.generate(input_ids_e_o)
    translated_text = english_to_odia_tokenizer.decode(translated_ids_e_o[0], skip_special_tokens=True)
    #translated_text = tokenizer.translate(source_text)
    return translated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        data = request.get_json()
        source_text = data['source_text']
        source_language = data['source_language']
        target_language = data['target_language']
        
        if source_language == 'Odia' and target_language == 'Hindi':
            translated_text = odia_to_hindi_translate(source_text)
        elif source_language == 'English' and target_language == 'Odia':
            translated_text = english_to_odia_translate(source_text)
        else:
            translated_text = "Translation not supported for the selected language pair"
        
        return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True)


