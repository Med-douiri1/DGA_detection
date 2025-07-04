import pandas as pd
import numpy as np
import string
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

# Same settings we used when training the model
max_len = 70
vocab = list(string.ascii_lowercase + string.digits + '-.')
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}

def encode_domain(domain):
    return [char_to_idx.get(c, 0) for c in domain.lower()]

def load_test_data(path):
    df = pd.read_csv(path)
    X = [encode_domain(d) for d in df['domain']]
    X = pad_sequences(X, maxlen=max_len)
    y_true = df['label'].values
    print(f" Loaded {len(df)} test samples.")
    return X, y_true

# Load our test data
X_test, y_true = load_test_data("test.csv")

# Load the model we trained earlier
model = load_model("model.h5")

# Run predictions on our test data
y_probs = model.predict(X_test, batch_size=128, verbose=1)
y_pred = np.argmax(y_probs, axis=1)

# Classification report 
report = classification_report(y_true, y_pred, digits=4)
print("\nClassification Report:")
print(report)

# Save report to file
with open("classification_report.txt", "w") as f:
    f.write(report)
