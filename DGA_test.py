from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.preprocessing import sequence
import numpy as np
import string
from sklearn.metrics import classification_report, confusion_matrix

# Parameters
max_len = 70

# Build character vocabulary
vocab = list(string.ascii_lowercase + string.digits + '-.')
char_to_idx = {char: i+1 for i, char in enumerate(vocab)}

def encode_domain(domain):
    return [char_to_idx.get(c, 0) for c in domain.lower()]

# Load the model
model = load_model('DGA_Model')

# Load new domains file (must have 'domain' and 'label' columns)
df_new = pd.read_csv('test_set.csv')
domains_new = df_new['domain'].astype(str).tolist()
labels_true = df_new['label'].astype(int).tolist()  

# Encode and pad
X_new = [encode_domain(d) for d in domains_new]
X_new = sequence.pad_sequences(X_new, maxlen=max_len)

# Predict
y_pred_probs = model.predict(X_new)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Metrics
print("\nClassification Report:")
print(classification_report(labels_true, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(labels_true, y_pred))
