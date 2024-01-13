import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score

df = pd.read_csv('Bharat_intern\Task1\SMSSpamCollection.txt',sep='\t',names=['label','message'])

print(df.head())
df['label'] = df['label'].map( {'spam': 1, 'ham': 0} )
X = df['message'].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(X_train)
encoded_train = tokeniser.texts_to_sequences(X_train)
encoded_test = tokeniser.texts_to_sequences(X_test)
max_length = 10
pad_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
pad_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')

vocab_size = len(tokeniser.word_index)+1

# defining the model


model=tf.keras.models.Sequential([
   tf.keras.layers.Embedding(input_dim=vocab_size,output_dim= 24, input_length=max_length),
   tf.keras.layers.SimpleRNN(24, return_sequences=False),
   tf.keras.layers.Dense(64, activation='relu'),
   tf.keras.layers.Dense(32, activation='relu'),
   tf.keras.layers.Dense(1, activation='sigmoid')
])

# compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)

model.fit(x=pad_train,
         y=y_train,
         epochs=5,
         validation_data=(pad_test, y_test),
         callbacks=[early_stop]
         )

def acc(y_true, y_pred):
   acc_sc = accuracy_score(y_true, y_pred)
   print(f"Accuracy : {str(round(acc_sc,2)*100)}")
   return acc_sc


preds = (model.predict(pad_test) > 0.5).astype("int32")
acc=accuracy_score(y_test, preds)
print(f"Accuracy : {str(round(acc,2)*100)}")
#Obtained an accuracy of 98.56%