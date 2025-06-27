import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# ✅ 1. Prepare Data
data = [
    ("hello", "bonjour"),
    ("how are you", "comment ça va"),
    ("thank you", "merci"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("see you later", "à plus tard"),
    ("I love you", "je t'aime"),
    ("what is your name", "comment tu t'appelles"),
    ("my name is", "je m'appelle"),
    ("where are you from", "d'où viens-tu"),
    ("I am from", "je viens de"),
    ("how old are you", "quel âge as-tu"),
    ("I am 25 years old", "j'ai 25 ans"),
    ("excuse me", "excusez-moi"),
    ("please", "s'il vous plaît"),
    ("I'm sorry", "je suis désolé"),
    ("can you help me", "pouvez-vous m'aider"),
    ("where is the bathroom", "où sont les toilettes"),
    ("I don't understand", "je ne comprends pas"),
    ("do you speak English", "parlez-vous anglais"),
    ("yes", "oui"),
    ("no", "non"),
    ("maybe", "peut-être"),
    ("I don't know", "je ne sais pas"),
    ("what time is it", "quelle heure est-il"),
    ("it's 3 o'clock", "il est trois heures"),
    ("goodbye", "au revoir"),
    ("see you tomorrow", "à demain"),
    ("I am tired", "je suis fatigué"),
    ("I am hungry", "j'ai faim"),
    ("I am thirsty", "j'ai soif"),
    ("let's go", "allons-y"),
    ("where is the train station", "où est la gare"),
    ("how much is this", "combien ça coûte"),
    ("I would like", "je voudrais"),
    ("do you have", "avez-vous"),
    ("I'm lost", "je suis perdu"),
    ("what's this", "qu'est-ce que c'est"),
    ("I'm looking for", "je cherche"),
    ("can I have the bill", "l'addition, s'il vous plaît"),
    ("what do you recommend", "que recommandez-vous"),
    ("I am allergic to", "je suis allergique à"),
    ("I need a doctor", "j'ai besoin d'un médecin"),
    ("I am sick", "je suis malade"),
    ("I am cold", "j'ai froid"),
    ("I am hot", "j'ai chaud"),
    ("congratulations", "félicitations"),
    ("happy birthday", "joyeux anniversaire"),
    ("merry Christmas", "joyeux Noël"),
    ("happy New Year", "bonne année"),
    ("can I take a picture", "puis-je prendre une photo"),
    ("where is the nearest hotel", "où est l'hôtel le plus proche"),
    ("do you have wifi", "avez-vous du wifi"),
    ("what's the password", "quel est le mot de passe"),
    ("I need a taxi", "j'ai besoin d'un taxi"),
    ("can I pay with card", "puis-je payer par carte"),
    ("can I pay in cash", "puis-je payer en espèces"),
    ("where is the restaurant", "où est le restaurant"),
    ("I don't eat meat", "je ne mange pas de viande"),
    ("I'm vegetarian", "je suis végétarien"),
    ("I'm vegan", "je suis végétalien"),
    ("I'm allergic to gluten", "je suis allergique au gluten"),
    ("what's your favorite food", "quel est ton plat préféré"),
    ("do you like it", "est-ce que tu aimes ça"),
    ("I like it", "j'aime ça"),
    ("I don't like it", "je n'aime pas ça"),
    ("it's delicious", "c'est délicieux"),
    ("it's spicy", "c'est épicé"),
    ("it's sweet", "c'est sucré"),
    ("it's salty", "c'est salé"),
    ("how far is it", "c'est loin"),
    ("can I get a map", "puis-je avoir une carte"),
    ("I need directions", "j'ai besoin de directions"),
    ("how do I get there", "comment y aller"),
    ("is it safe", "est-ce sûr"),
    ("I love traveling", "j'adore voyager"),
    ("I'm here for work", "je suis ici pour le travail"),
    ("I'm here for vacation", "je suis ici en vacances"),
    ("where is the beach", "où est la plage"),
    ("what's the weather like", "quel temps fait-il"),
    ("it's sunny", "il fait soleil"),
    ("it's raining", "il pleut"),
    ("it's cold", "il fait froid"),
    ("it's warm", "il fait chaud"),
    ("do you have a room", "avez-vous une chambre"),
    ("can I have the menu", "puis-je avoir le menu"),
    ("what's your favorite movie", "quel est ton film préféré"),
    ("I like reading", "j'aime lire"),
    ("I like listening to music", "j'aime écouter de la musique"),
    ("I like watching movies", "j'aime regarder des films"),
    ("what's your job", "quel est ton travail"),
    ("I work as", "je travaille comme"),
    ("I'm a student", "je suis étudiant"),
    ("where do you live", "où habites-tu"),
    ("I live in", "j'habite à"),
    ("how long have you been here", "depuis combien de temps es-tu ici"),
    ("I have been here for two weeks", "je suis ici depuis deux semaines"),
    ("I'm going home", "je rentre à la maison"),
    ("what are you doing", "que fais-tu"),
    ("I'm learning French", "j'apprends le français"),
    ("do you like France", "aimes-tu la France"),
    ("I love France", "j'adore la France"),
    ("I want to visit Paris", "je veux visiter Paris"),
    ("how can I help you", "comment puis-je vous aider"),
    ("I need help", "j'ai besoin d'aide"),
    ("what time does it open", "à quelle heure ça ouvre"),
    ("what time does it close", "à quelle heure ça ferme"),
    ("can I sit here", "puis-je m'asseoir ici"),
    ("can you repeat that", "pouvez-vous répéter cela"),
    ("slow down", "ralentis"),
    ("where can I buy tickets", "où puis-je acheter des billets"),
    ("do you like sports", "aimes-tu le sport"),
    ("I like football", "j'aime le football"),
    ("I like basketball", "j'aime le basketball"),
    ("let's play", "jouons"),
    ("what time is the game", "à quelle heure est le match"),
    ("how do you say", "comment dit-on"),
    ("can you show me", "pouvez-vous me montrer"),
    ("can you write it down", "pouvez-vous l'écrire"),
    ("I need a break", "j'ai besoin d'une pause"),
    ("I'm busy", "je suis occupé"),
    ("I'm free", "je suis libre"),
    ("what's happening", "que se passe-t-il"),
    ("I like dancing", "j'aime danser"),
    ("what's your phone number", "quel est ton numéro de téléphone"),
    ("I'll call you", "je t'appellerai"),
    ("I miss you", "tu me manques"),
    ("see you soon", "à bientôt"),
    ("what do you think", "qu'en penses-tu"),
    ("I agree", "je suis d'accord"),
    ("I disagree", "je ne suis pas d'accord"),
    ("can you wait", "peux-tu attendre"),
    ("I don't have time", "je n'ai pas le temps"),
    ("hurry up", "dépêche-toi"),
    ("are you ready", "es-tu prêt"),
    ("what's wrong", "qu'est-ce qui ne va pas"),
    ("nothing", "rien"),
    ("everything's fine", "tout va bien"),
    ("I'm happy", "je suis heureux"),
    ("I'm sad", "je suis triste"),
    ("I'm angry", "je suis en colère"),
    ("I'm excited", "je suis excité"),
    ("it's okay", "c'est d'accord"),
    ("let's meet", "rencontrons-nous"),
    ("I'm proud of you", "je suis fier de toi"),
    ("let's celebrate", "célébrons"),
    ("where is the nearest pharmacy", "où est la pharmacie la plus proche"),
    ("what's your address", "quelle est ton adresse"),
    ("I'm cold", "j'ai froid"),
    ("do you need anything", "as-tu besoin de quelque chose"),
    ("what's the price", "quel est le prix"),
    ("that's expensive", "c'est cher"),
    ("that's cheap", "c'est bon marché"),
    ("I like your style", "j'aime ton style"),
    ("I'm looking forward to it", "j'ai hâte"),
    ("it's beautiful", "c'est beau"),
    ("it's ugly", "c'est moche"),
    ("I'm nervous", "je suis nerveux"),
    ("what's your favorite color", "quelle est ta couleur préférée"),
    ("I need more time", "j'ai besoin de plus de temps"),
    ("I don't have enough money", "je n'ai pas assez d'argent"),
    ("where can I exchange money", "où puis-je échanger de l'argent"),
    ("can you recommend a hotel", "pouvez-vous recommander un hôtel"),
    ("do you have any plans", "as-tu des projets"),
    ("what are your hobbies", "quels sont tes passe-temps"),
    ("can I borrow this", "puis-je emprunter ceci"),
    ("are you free tomorrow", "es-tu libre demain"),
    ("I'm not sure", "je ne suis pas sûr"),
    ("let's keep in touch", "restons en contact"),
    ("can you hear me", "m'entends-tu"),
    ("take care", "prends soin de toi"),
    ("I need to go", "je dois y aller"),
    ("what a surprise", "quelle surprise"),
    ("that's interesting", "c'est intéressant"),
    ("that's boring", "c'est ennuyeux"),
    ("I'm in a hurry", "je suis pressé")
]


# English to French
eng_sentences = [pair[0].lower() for pair in data]
fr_sentences = ['<start> ' + pair[1].lower() + ' <end>' for pair in data]

#  2. Tokenization
tokenizer_eng = Tokenizer(filters='', lower=True)
tokenizer_eng.fit_on_texts(eng_sentences)
input_sequences = tokenizer_eng.texts_to_sequences(eng_sentences)
max_encoder_len = max(len(seq) for seq in input_sequences)
encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_len, padding='post')

tokenizer_fr = Tokenizer(filters='', lower=True)
tokenizer_fr.fit_on_texts(fr_sentences)
target_sequences = tokenizer_fr.texts_to_sequences(fr_sentences)
max_decoder_len = max(len(seq) for seq in target_sequences)
decoder_input_data = pad_sequences([seq[:-1] for seq in target_sequences], maxlen=max_decoder_len - 1, padding='post')
decoder_target_data = pad_sequences([seq[1:] for seq in target_sequences], maxlen=max_decoder_len - 1, padding='post')

vocab_input_size = len(tokenizer_eng.word_index) + 1
vocab_target_size = len(tokenizer_fr.word_index) + 1
decoder_target_data_oh = tf.keras.utils.to_categorical(decoder_target_data, num_classes=vocab_target_size)

#  3. Build Model
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding_layer = Embedding(input_dim=vocab_input_size, output_dim=latent_dim, mask_zero=True)
encoder_emb = encoder_embedding_layer(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding_layer = Embedding(input_dim=vocab_target_size, output_dim=latent_dim, mask_zero=True)
decoder_emb = decoder_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_target_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#  4. Train Model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data_oh, batch_size=16, epochs=300, verbose=1)

#  5. Save Model & Tokenizers
os.makedirs("model", exist_ok=True)
model.save("model/translation_model.keras")

# Inference Encoder
encoder_model = Model(encoder_inputs, encoder_states)

# Inference Decoder
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inf_emb = decoder_embedding_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_inf_emb, initial_state=decoder_states_inputs)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2, state_h2, state_c2])

encoder_model.save("model/encoder_model.keras")
decoder_model.save("model/decoder_model.keras")

#  Save Tokenizers and Word Index
with open("model/tokenizer_eng.pkl", "wb") as f:
    pickle.dump(tokenizer_eng, f)
with open("model/tokenizer_fr.pkl", "wb") as f:
    pickle.dump(tokenizer_fr, f)

reverse_target_word_index = {i: w for w, i in tokenizer_fr.word_index.items()}
with open("model/reverse_target_word_index.pkl", "wb") as f:
    pickle.dump(reverse_target_word_index, f)

print(" Training complete")
print("max_input_len =", max_encoder_len)
print("max_target_len =", max_decoder_len - 1)
