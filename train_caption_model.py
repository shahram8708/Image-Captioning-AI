import os
import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

IMAGES_DIR = 'path/to/images'
CAPTIONS_FILE = 'path/to/captions.txt'

MAX_LENGTH = 34
VOCAB_SIZE = 5000

def extract_features(directory):
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    features = {}
    for img_name in os.listdir(directory):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(directory, img_name)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            features[img_name] = feature
    return features

def load_captions(filepath):
    with open(filepath, 'r') as file:
        captions = file.readlines()
    return {line.split()[0]: line.strip().split()[1:] for line in captions}

def preprocess_captions(captions):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return sequences, tokenizer

def data_generator(captions, features, tokenizer, max_length):
    while True:
        for key, desc_list in captions.items():
            feature = features[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=VOCAB_SIZE)[0]
                    yield [[feature, in_seq], out_seq]

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def main():
    features = extract_features(IMAGES_DIR)
    captions = load_captions(CAPTIONS_FILE)
    sequences, tokenizer = preprocess_captions(captions)
    model = define_model(VOCAB_SIZE, MAX_LENGTH)
    generator = data_generator(captions, features, tokenizer, MAX_LENGTH)
    model.fit(generator, epochs=20, steps_per_epoch=1000, verbose=2, callbacks=[ModelCheckpoint('caption_model.h5', save_best_only=True)])
    print("Model training complete and saved as caption_model.h5")

if __name__ == '__main__':
    main()
