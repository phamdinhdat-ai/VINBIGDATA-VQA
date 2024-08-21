import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import pandas as pd 
import json
import torch
from collections import Counter
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string



# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def segment_text(text):
    result = word_tokenize(text)
    return result

def find_most_common_answer(answers):
    answer_counter = Counter(answers)
    most_common_answers = answer_counter.most_common()
    most_common_answer, count = most_common_answers[0]
    return most_common_answer

def select_most_common_answers(df):
    selected_answers = []
    for idx, row in df.iterrows():
        answers = [answer["answer"] for answer in row["answers"]]
        selected_answer = find_most_common_answer(answers)
        
        selected_answers.append({
            "answer": selected_answer
        })

    # Update the "answer" and "answer_confidence" columns in the DataFrame
    df[["answer"]] = pd.DataFrame(selected_answers)
    df["answers"] = df["answers"].apply(lambda x: [answer["answer"] for answer in x])
    
    return df



def plot_loss(train_loss, val_loss):
    epochs = range(1, len(train_loss + 1))
    
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, val_loss, label='Val')
    plt.title("Training progress ....")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
def plot_img(img_path):
    image = mpimg.imread(img_path)
    plt.imshow(image)
    plt.axis('off')
    
    

    

    