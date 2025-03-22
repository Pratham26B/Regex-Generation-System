import pandas as pd
import regex as re
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
import spacy
import os

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# PHASE 1: LEARNING PHASE (TRAINING ON EXISTING REGEX-DATA PAIRS)

def load_regex_data(file_paths):
    regex_data = {}  # Dictionary to store regex as key and comment as value

    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or "#" not in line:  # Ignore empty lines and lines without comments
                    continue

                # Extract regex pattern and comment
                pattern, comment = re.split(r'\s*#\s*', line, maxsplit=1)
                pattern = pattern.strip()
                comment = comment.strip()

                if pattern.startswith("r'") and pattern.endswith("'"):
                    pattern = pattern[2:-1].rstrip(',')  # Remove r' and trailing commas
                    regex_data[pattern] = comment  # Store in dictionary

    if not regex_data:
        raise ValueError("No valid regex patterns found in the provided files. Check file format.")

    return regex_data

# Tokenization function
def tokenize_regex_patterns(regex_patterns):
    if not regex_patterns:
        raise ValueError("Empty regex patterns list. Ensure data is correctly loaded.")

    unique_chars = sorted(set(''.join(regex_patterns.keys())))
    char_to_index = {char: i+1 for i, char in enumerate(unique_chars)}
    index_to_char = {i: char for char, i in char_to_index.items()}

    tokenized_patterns = [[char_to_index[c] for c in pattern] for pattern in regex_patterns.keys()]
    max_length = max(len(seq) for seq in tokenized_patterns) if tokenized_patterns else 0
    tokenized_patterns = tf.keras.preprocessing.sequence.pad_sequences(tokenized_patterns, maxlen=max_length, padding='post')

    return tokenized_patterns, char_to_index, index_to_char, max_length

# Model definition
def build_lstm_model(vocab_size, max_length):
    encoder_inputs = Input(shape=(max_length,))
    encoder_embedding = Embedding(vocab_size, 128, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

    decoder_inputs = Input(shape=(max_length,))
    decoder_embedding = Embedding(vocab_size, 128, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
    decoder_dense = Dense(vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model

file_paths = ['regexpatternslist.txt']  # Add other files if needed
regex_data = load_regex_data(file_paths)
tokenized_patterns, char_to_index, index_to_char, max_length = tokenize_regex_patterns(regex_data)
X_train, X_test = train_test_split(tokenized_patterns, test_size=0.1, random_state=42)
decoder_input_data = np.hstack([np.zeros((X_train.shape[0], 1)), X_train[:, :-1]])
y_train = np.expand_dims(X_train, axis=-1)

model = build_lstm_model(len(char_to_index) + 1, max_length)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model.fit([X_train, decoder_input_data], y_train, batch_size=32, epochs=50, validation_split=0.1, callbacks=[early_stopping])
model.save("regex_lstm_model.keras", save_format="keras")
print("Model training complete and saved!")

# PHASE 2: GENERATION PHASE (PREDICT & REFINE NEW REGEX)

model = load_model("regex_lstm_model.keras")
index_to_char = {i: c for c, i in char_to_index.items()}


def sequence_to_regex(sequence):
    regex_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.*+?^${}()|[]\\")
    regex_pattern = ''.join(index_to_char[i] for i in sequence if i in index_to_char and index_to_char[i] in regex_chars)

    # Extract strictly between `^` and `$`
    match = re.search(r'\^.*?\$', regex_pattern)  # Extracts the valid part
    if match:
        regex_pattern = match.group(0)  # Keep only the portion between `^` and `$`
    else:
        regex_pattern = "^.*$"  # Default fallback pattern

    return regex_pattern


def predict_regex_pattern(values):
    tokenized_values = [[char_to_index.get(c, 0) for c in v] for v in values]
    tokenized_values = tf.keras.preprocessing.sequence.pad_sequences(tokenized_values, maxlen=max_length, padding='post')
    decoder_input_data = np.hstack([np.zeros((tokenized_values.shape[0], 1)), tokenized_values[:, :-1]])

    predictions = model.predict([tokenized_values, decoder_input_data])
    predicted_regex = np.argmax(predictions, axis=-1)

    refined_regex = sequence_to_regex(predicted_regex[0])

    # Ensure regex isn't None or empty
    if not refined_regex or refined_regex.strip() == "" or len(refined_regex) < 5:
        print(f"⚠️ Warning: Empty or too short regex predicted for values: {values}")
        return ".*"  # Return a match-all pattern instead of None

    return refined_regex


def refine_regex_pattern(values, predicted_regex):
    extracted_pattern = extract_regex_pattern(values)

    # Ensure both regex patterns are valid strings
    regex_patterns = [pattern for pattern in [predicted_regex, extracted_pattern] if pattern]

    if not regex_patterns:  # If no valid regex is found, return a fallback pattern
        return ".*"  # Generic match-all pattern

    return "|".join(set(regex_patterns))

def detect_pattern_type(value):
    """
    Detects if a value matches a predefined structured format (email, phone, date, etc.).
    """
    PREDEFINED_PATTERNS = {
        "email": r"^[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}$",
        "phone": r"^(\+\d{1,3}[- ]?)?\d{10}$",
        "date": r"^(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})$",
        "amount": r"^\$?\d{1,3}(,\d{3})*(\.\d{2})?$"
    }

    for pattern_name, pattern in PREDEFINED_PATTERNS.items():
        if re.match(pattern, value):
            return pattern
    return None

def generate_generalized_pattern(values):
    """
    Generates a generalized regex pattern based on sample values.
    """
    char_classes = []
    max_length = max(len(v) for v in values)

    for i in range(max_length):
        chars_at_pos = {v[i] if i < len(v) else "" for v in values}

        if all(c.isdigit() for c in chars_at_pos if c):
            char_classes.append("\\d+")
        elif all(c.isalpha() for c in chars_at_pos if c):
            char_classes.append("[a-zA-Z]+")
        elif len(chars_at_pos) == 1:
            char_classes.append(re.escape(chars_at_pos.pop()))
        else:
            char_classes.append(".")

    return "^" + "".join(char_classes) + "$"

def extract_regex_pattern(values):
    detected_patterns = [detect_pattern_type(value) for value in values]
    filtered_patterns = [pattern for pattern in detected_patterns if pattern]
    if filtered_patterns:
        return Counter(filtered_patterns).most_common(1)[0][0]
    return generate_generalized_pattern(values)


def clean_regex(regex_pattern):
    """
    Cleans the regex pattern:
    - Removes '.*|' if it appears at the start
    - Removes '|.*' if it appears at the end
    - Adds 'r' before the '^' sign
    """
    regex_pattern = regex_pattern.strip()  # Remove unnecessary spaces
    
    # Remove '.*|' from the beginning
    regex_pattern = re.sub(r'^\.\*\|', '', regex_pattern)
    
    # Remove '|.*' from the end
    regex_pattern = re.sub(r'\|\.\*$', '', regex_pattern)

    # Ensure 'r' is added before '^'
    if regex_pattern.startswith("^"):
        regex_pattern = "r" + regex_pattern

    return regex_pattern


def is_valid_regex(pattern):
    try:
        re.compile(pattern)  # Try compiling the regex
        return True
    except re.error as e:
        print(f"❌ Invalid Regex: {pattern}\n🚨 Error: {e}\n")
        return False


temp_regex_dict = {}

def generate_regex_for_columns(file_path):
    global temp_regex_dict
    file_extension = os.path.splitext(file_path)[-1].lower()
    regex_patterns = {}
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        sheet_name = input("Enter the sheet name to process: ")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    

    temp_regex_dict.clear()  # Clear previous values

    for col in df.columns:
        sample_values = df[col].dropna().astype(str).sample(min(10, len(df[col]))).tolist()
        predicted_regex = predict_regex_pattern(sample_values)
        refined_regex = refine_regex_pattern(sample_values, predicted_regex)
        cleaned_regex = clean_regex(refined_regex)
        if is_valid_regex(cleaned_regex):
            regex_patterns[col] = cleaned_regex
            temp_regex_dict[col] = cleaned_regex
    
    print("Temporary Dictionary of Column Regex Patterns:")
    print(temp_regex_dict)
    
    with open("updated_regex_patterns.txt", "w", encoding="utf-8") as f:
        for col, pattern in regex_patterns.items():
            f.write(f"Column: {col}, Refined Regex: {pattern}\n")
    
    return regex_patterns


file_path = "Stock-Items.csv"
regex_dict = generate_regex_for_columns(file_path)


# Print temp_regex_dict outside function
print("Accessing temp_regex_dict outside the function:")
print(temp_regex_dict)
