import json
import pickle
import numpy as np
import random
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model  # type: ignore

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

#  Clean and tokenize sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#  Create bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#  Predict intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

# Get chatbot response
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that. Could you rephrase?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']  #  Fixed typo here
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])  #  Fixed key: 'response' to 'responses'
            return result
    return "Sorry, I don't have a response for that right now."

#  Chat loop
print("Hello. How may I help you? (Type 'quit' to exit)")

while True:
    message = input("")
    if message.lower() == "quit":
        print("Goodbye! 👋")
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
