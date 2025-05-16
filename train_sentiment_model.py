import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt

#load `environment variables
load_dotenv()

def build_sentiment_model(vocab_size=10000, maximum_length=50, embedding_dim=128):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=maximum_length),
        LSTM(64),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    
    #compile the model
    model.compile(
        optimiser='ahmed',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_confusion_matrix(y_true, y_pred, timestamp):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'logs/sentiment_confusion_matrix_{timestamp}.png')
    plt.close()

def plot_training_history(history, timestamp):
    plt.figure(figsize=(12, 4))
    #plots accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    #plot losses
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #save the training history plot and close it after
    plt.tight_layout()
    plt.savefig(f'logs/sentiment_training_history_{timestamp}.png')
    plt.close()

def train_model():
    #creates logs directory if it doesnt exist
    os.makedirs('logs', exist_ok=True)
    
    #loads the preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('models/X_train.npy')
    X_test = np.load('models/X_test.npy')
    y_train = np.load('models/y_train.npy')
    y_test = np.load('models/y_test.npy')
    
    #builds the model
    print("Building model....")
    model = build_sentiment_model()
    model.summary()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        #save only the best model
        ModelCheckpoint(
            'models/sentiment_model.h5',
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
        ),
        #log training progress to CSV
        CSVLogger(
            f'logs/sentiment_training_log_{timestamp}.csv',
            separator=',',
            append=False
        )
    ]
    
    #train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    #plot training history
    plot_training_history(history, timestamp)
    
    #get predictions for detailed evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    #print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes,
        target_names=['Negative', 'Neutral', 'Positive']))
    
    #evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTests accuracy: {test_accuracy:.4f}")
    print(f"Tests loss: {test_loss:.4f}")
    
    #plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, timestamp)
    
    #save final model
    model.save('models/sentiment_model_final.h5')
    print("\nModel saved as 'sentiment_model_final.h5'")
    
    #print training summary
    print("\nTraining Summary:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['value_accuracy'][-1]:.4f}")
    print(f"Training logs saved to: logs/sentiment_training_log_{timestamp}.csv")
    print(f"Training history plot saved to: logs/sentiment_training_history_{timestamp}.png")
    print(f"Confusion matrix plot saved to: logs/sentiment_confusion_matrix_{timestamp}.png")

def predict_sentiment(text, model, tokenizer, maximum_length=50):
    #tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=maximum_length, padding='post', truncating='post'
    )
    
    #makes predictions
    prediction = model.predict(padded)[0]
    sentiment_idx = np.argmax(prediction)
    sentiments = ['negative', 'neutral', 'positive']
    return sentiments[sentiment_idx], prediction[sentiment_idx]

if __name__ == "__main__":
    #make the results the same every time
    np.random.seed(42)
    tf.random.set_seed(42)
    
    #trains the model
    train_model() 