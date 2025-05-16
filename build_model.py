import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from dotenv import load_dotenv

#load dotenv::environment variables
load_dotenv()

def build_model(vocab_size=10000, maximum_length=50, embedding_dim=128):
    model = Sequential([
        
        Embedding(vocab_size, embedding_dim, input_length=maximum_length),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    #compile the model
    model.compile(
        optimiser='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    #load preprocesed data
    print("Loading preprocessed data....")
    X_train = np.load('models/X_train.npy')
    X_test = np.load('models/X_test.npy')
    y_train = np.load('models/y_train.npy')
    y_test = np.load('models/y_test.npy')
    
    #build the model
    print("Building model../.")
    model = build_model()
    model.summary()
    
    #define any callbacks
    callbacks = [
        #save best only
        ModelCheckpoint(
            'models/toxicity_model.h5',
            monitor='value_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        #early stopping to prevent overfitting
        EarlyStopping(
            monitor='value_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    #train the model
    print("\nTraining model....")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    #evaluate the model
    print("\nEvaluating model....")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    #make the results the same every time
    np.random.seed(42)
    tf.random.set_seed(42)
    
    #train model
    train_model() 