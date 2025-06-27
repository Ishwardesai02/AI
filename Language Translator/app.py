import numpy as np
import tensorflow as tf
import pickle
from flask import Flask, render_template, request

#  Load models in .keras format
encoder_model = tf.keras.models.load_model("model/encoder_model.keras", compile=False)
decoder_model = tf.keras.models.load_model("model/decoder_model.keras", compile=False)

#  Load tokenizers and reverse index
with open("model/tokenizer_eng.pkl", "rb") as f:
    tokenizer_eng = pickle.load(f)
with open("model/tokenizer_fr.pkl", "rb") as f:
    tokenizer_fr = pickle.load(f)
with open("model/reverse_target_word_index.pkl", "rb") as f:
    reverse_target_word_index = pickle.load(f)

#  These must match values from 1.py
max_encoder_len = 7
max_decoder_len = 8
latent_dim = 256

app = Flask(__name__)

#  Translation function
def translate_sentence(input_text):
    input_seq = tokenizer_eng.texts_to_sequences([input_text.lower()])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_encoder_len, padding='post')
    
    # Get encoder states
    states_value = encoder_model.predict(input_seq)

    # Start token for French
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_fr.word_index['<start>']

    decoded_sentence = ""
    stop_condition = False

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_decoder_len:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        # Update for next timestep
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/translate', methods=['POST'])
def translate():
    input_text = request.form['sentence']
    try:
        output_text = translate_sentence(input_text)
    except Exception as e:
        output_text = f"Translation Error: {e}"
    return render_template("index.html", input_text=input_text, output_text=output_text)

if __name__ == '__main__':
    app.run(debug=True)
