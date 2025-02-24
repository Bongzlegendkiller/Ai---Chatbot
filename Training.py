import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Activation, Dropout  # type: ignore
from tensorflow.keras.optimizers import SGD  # type: ignore

# Download necessary NLTK data
nltk.download('punkt')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each intent pattern
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # Add all words to the words list
        documents.append((word_list, intent['tag']))  # Store the pattern with its tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean words list
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save processed data
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)

for document in documents:
    # Create a bag of words
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Check for each word in the vocabulary
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Create the output row for the current pattern
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    # Corrected this line by adding a tuple (bag, output_row)
    training.append((bag, output_row))

# Shuffle the training data
random.shuffle(training)

# Convert list of tuples into two separate lists
train_x = [t[0] for t in training]
train_y = [t[1] for t in training]

# Convert to numpy arrays if needed
train_x = np.array(train_x)
train_y = np.array(train_y)


# Model Architecture
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))  # ✅ Fixed 'softmax' typo

# Optimizer Configuration
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # ✅ Updated parameters

# Compile the Model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # ✅ Fixed loss function typo

# Model Training
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save Model
model.save('chatbot_model.h5', hist)  # ✅ Updated filename to avoid spaces
print("Model training complete and saved successfully.")

