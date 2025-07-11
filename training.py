import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# 🎯 Extract MFCC features from audio
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 📂 Load TESS audio files from flat folder
def load_tess_data(data_path):
    features = []
    for file in os.listdir(data_path):
        if file.endswith(".wav"):
            try:
                emotion = file.split("_")[-1].replace(".wav", "").lower()
                file_path = os.path.join(data_path, file)
                data = extract_features(file_path)
                features.append([data, emotion])
            except Exception as e:
                print(f"Error processing {file}: {e}")
    return pd.DataFrame(features, columns=["feature", "emotion"])

# 📥 Load dataset
data = load_tess_data("tess_data")  # Replace with your actual folder name

# 🧾 Check the loaded data
print(f"\nLoaded {len(data)} samples.")
print("Emotion label distribution:\n", data["emotion"].value_counts())

# 🧠 Prepare data
X = np.array(data["feature"].tolist())
y = np.array(data["emotion"].tolist())

# 🔢 Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 🧪 Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# 🤖 Build the neural network model
model = Sequential()
model.add(Dense(256, input_shape=(40,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🏋️ Train the model
print("\nTraining model...\n")
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# ✅ Evaluate the model
print("\nEvaluating model...\n")
y_pred = model.predict(X_test)
y_pred_labels = le.inverse_transform(np.argmax(y_pred, axis=1))
y_true_labels = le.inverse_transform(np.argmax(y_test, axis=1))

print("Classification Report:\n")
print(classification_report(y_true_labels, y_pred_labels))
