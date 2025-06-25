from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
import string
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Parameters 
max_features = 40
max_len = 70
batch_size = 128
epochs = 100

# Build the character vocabulary 
vocab = list(string.ascii_lowercase + string.digits + '-.')
char_to_idx = {char: i+1 for i, char in enumerate(vocab)}
vocab_size = len(char_to_idx) + 1

def encode_domain(domain):
    return [char_to_idx.get(c, 0) for c in domain]

# Load our training data 
df_train = pd.read_csv("train.csv")
domains_train = df_train['domain'].astype(str).str.lower().tolist()
labels_train = df_train['binary_label'].astype(int).tolist()

# Convert domains to numbers and pad them to same length
X_train = [encode_domain(d) for d in domains_train]
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
y_train = np.array(labels_train)

# Do the same for validation data
df_val = pd.read_csv("validation.csv")
domains_val = df_val['domain'].astype(str).str.lower().tolist()
labels_val = df_val['binary_label'].astype(int).tolist()
X_val = [encode_domain(d) for d in domains_val]
X_val = sequence.pad_sequences(X_val, maxlen=max_len)
y_val = np.array(labels_val)

# Build our Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Set up our optimizer and compile the model
optimizer = Adam(learning_rate=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Custom callback to show a progress bar during training
class TQDMProgressBar(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_bar = tqdm(total=int(len(X_train) / batch_size), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    def on_batch_end(self, batch, logs=None):
        self.epoch_bar.update(1)
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.close()
        
# Stop training if we're not improving after 8 epochs
early_stop = EarlyStopping(patience=8, restore_best_weights=True)

# Reduce learning rate 
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-5
)

# Training the model
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[TQDMProgressBar(), early_stop, lr_scheduler]
)


# Save our trained model
model.save("model_final.h5")


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc



plt.figure(figsize=(12, 4))

# Plot loss over time
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy over time
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_accuracy_plot11.png")  
plt.show()

# Check how well we did on validation data
loss, acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {acc:.4f}")

# Get predictions and see detailed results
y_pred_probs = model.predict(X_val)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_val, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))


# Create ROC curve to see how good our classifier is
fpr, tpr, _ = roc_curve(y_val, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Validation ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)

plt.savefig("roc_curve11.png")  
plt.show()

# Now test on unseen data
df_test = pd.read_csv("test.csv")
domains_test = df_test['domain'].astype(str).str.lower().tolist()
labels_test = df_test['binary_label'].astype(int).tolist()
X_test = [encode_domain(d) for d in domains_test]
X_test = sequence.pad_sequences(X_test, maxlen=max_len)
y_test = np.array(labels_test)

# Final evaluation on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Get final predictions and detailed results
y_test_pred_probs = model.predict(X_test)
y_test_pred = (y_test_pred_probs > 0.5).astype(int).flatten()

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred, digits=4))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
