import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from dotenv import load_dotenv

#load dotenv::environment variables
load_dotenv()

def load_and_preprocess_data(file_path, max_words=10000, maximum_length=50):
    #loads the dataset
    print("Loading dataset....")
    df = pd.read_csv(file_path)
    
    #mark rows as toxic
    df['toxic'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0).astype(int)
    
    #get the text and labels
    texts = df['comment_text'].values
    labels = df['toxic'].values
    
    #convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    #print sequence length statistics
    seq_lengths = [len(seq) for seq in sequences]
    print(f"\nSequence length statistics:")
    print(f"Min length: {min(seq_lengths)}")
    print(f"Max length: {max(seq_lengths)}")
    print(f"Mean length: {sum(seq_lengths)/len(seq_lengths):.2f}")
    
    #print binary toxicity distribution
    toxic_count = df['toxic'].sum()
    toxic_percentage = (toxic_count / len(df)) * 100
    print(f"\nBinary toxicity: {toxic_count} toxic comments ({toxic_percentage:.2f}%)")
    
    #pad sequences
    print("\nPadding sequences...")
    padded_sequences = pad_sequences(sequences, maxlen=maximum_length, padding='post', truncating='post')
    
    #split the data into training and testing sets
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
        #print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total comments: {len(df)}")
    print("\nToxicity distribution:")
    for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        print(f"{col}: {count} ({percentage:.2f}%)")
    
        #set up and tokenize the text
    print("\nTokenizing text....")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    
    #print split statistics
    print("\nSplit Statistics:")
    print(f"Training set: {len(X_train)} samples ({sum(y_train)} toxic)")
    print(f"Testing set: {len(X_test)} samples ({sum(y_test)} toxic)")
    
    return X_train, X_test, y_train, y_test, tokenizer

def save_tokenizer(tokenizer, save_path):
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(tokenizer, f)

def main():
    #create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    #load the data
    X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess_data(
        'data/jigsaw-toxic-comment-train.csv'
    )
    
    #save the tokenizer
    save_tokenizer(tokenizer, 'models/tokenizer.pkl')
    
    
    print("\nData preprocessing completed!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print("\nProcessed data and tokenizer have been saved in the 'models' directory.")
    
    #save the data
    np.save('models/X_train.npy', X_train)
    np.save('models/X_test.npy', X_test)
    np.save('models/y_train.npy', y_train)
    np.save('models/y_test.npy', y_test)

if __name__ == "__main__":
    main() 