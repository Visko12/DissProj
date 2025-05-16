import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import os
from datetime import datetime
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_class_weight

#load dotenv::environment
load_dotenv()

def build_model(vocab_size=10000, maximum_length=50, embedding_dim=128):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=maximum_length),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    #compile the model with a higher learning rate
    model.compile(
        optimiser=tf.keras.optimizers.Adam(learning_rate=0.002),  # Increased learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    os.makedirs('logs', exist_ok=True)
    
    #load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('models/X_train.npy')
    X_test = np.load('models/X_test.npy')
    y_train = np.load('models/y_train.npy')
    y_test = np.load('models/y_test.npy')
    
    #calculate class weights with higher weight for toxic class
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    #increase weight for toxic class significantly
    class_weights[1] *= 8.0
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    #build the model
    print("Building model...")
    model = build_model()
    
    #print model summary
    model.summary()
    
    #create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #define callbacks
    callbacks = [
        #save the best model
        ModelCheckpoint(
            'models/toxicity_model.keras',
            monitor='value_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        #early stopping to prevent overfitting
        EarlyStopping(
            monitor='value_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        #reduce learning rate when plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='value_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        ),
        #log training progress to CSV
        CSVLogger(
            f'logs/training_log_{timestamp}.csv',
            separator=',',
            append=False
        )
    ]
    
    #train the model
    print("\nTraining model....")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,  #more epochs
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    #evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    #save final model
    model.save('models/toxicity_model_final.keras')
    print("\nModel saved as 'toxicity_model_final.keras'")
    
    #print training summary
    print("\nTraining Summary:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['value_accuracy'][-1]:.4f}")
    print(f"Training logs saved to: logs/training_log_{timestamp}.csv")

if __name__ == "__main__":
    #make the results the same every time
    np.random.seed(42)
    tf.random.set_seed(42)
    
    #train model
    train_model() 