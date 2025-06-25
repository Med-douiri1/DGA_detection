import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import RMSprop

# === Parameters ===
max_len = 70
embedding_dim = 128
lstm_units = 128
batch_size = 128
epochs = 50

# Create our character vocabulary
vocab = list(string.ascii_lowercase + string.digits + '-.')
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
vocab_size = len(char_to_idx) + 1  # +1 for padding

def encode_domain(domain):
    return [char_to_idx.get(c, 0) for c in domain.lower()]

# Helper function to load and prepare our data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = [encode_domain(d) for d in df['domain']]
    X = pad_sequences(X, maxlen=max_len)
    y = to_categorical(df['label'])
    return X, y

# Load our training and validation data
X_train, y_train = load_data("train.csv")
X_val, y_val = load_data("validation.csv")
nb_classes = y_train.shape[1]

# Build our model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(lstm_units))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

from keras.optimizers import Adam

# Set up the model for training - using Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Reduce learning rate
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
# Save the best model during training
checkpoint = ModelCheckpoint("model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# Train the  model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[TqdmCallback(verbose=1), lr_scheduler, checkpoint],
    verbose=0
)

# Save the model
model.save("model_final.h5")

# Plot Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')

# Plot Accuracy 
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

