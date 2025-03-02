#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer


# In[ ]:


nltk.download('punkt')
nltk.download('wordnet')


# In[ ]:


lemmatizer = WordNetLemmatizer()


# In[ ]:


# Define chatbot intents with different content
intent_data = [
    {"category": "motivation", "examples": ["I feel down", "I'm demotivated", "Help me stay positive"], "replies": ["You are capable of amazing things!", "Believe in yourself and all that you are.", "Every day is a new beginning."]},
    {"category": "study_tips", "examples": ["How can I focus on studies?", "Tips for studying", "Help me concentrate"], "replies": ["Break tasks into small steps and take breaks!", "Set a timer for focused work sessions.", "Create a study schedule and stick to it!"]},
    {"category": "time_management", "examples": ["How to manage time?", "I have too much to do", "Time management tips"], "replies": ["Prioritize tasks and make a to-do list.", "Use the Pomodoro technique for productivity.", "Avoid multitasking and stay focused."]},
    {"category": "self_care", "examples": ["How to take care of myself?", "What is self-care?", "I feel overwhelmed"], "replies": ["Remember to rest, hydrate, and breathe deeply.", "Self-care is not selfish, it's necessary.", "Take small breaks and do something you enjoy."]},
    {"category": "goals", "examples": ["How to set goals?", "I want to achieve something", "Goal setting tips"], "replies": ["Set SMART goals: Specific, Measurable, Achievable, Relevant, and Time-bound.", "Break big goals into smaller, manageable tasks.", "Track your progress and celebrate small wins."]}
]


# In[ ]:


sentences = []
labels = []
responses = {}


# In[ ]:


for item in intent_data:
    for example in item['examples']:
        sentences.append(example)
        labels.append(item['category'])
    responses[item['category']] = item['replies']


# In[ ]:


def preprocess(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return ' '.join(words)


# In[ ]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([preprocess(sentence) for sentence in sentences])


# In[ ]:


model = LogisticRegression()
model.fit(X, labels)


# In[1]:


def generate_response(user_input):
    input_vector = vectorizer.transform([preprocess(user_input)])
    predicted_label = model.predict(input_vector)[0]
    return random.choice(responses[predicted_label])


# In[ ]:


print("Chatbot is ready! Type 'exit' to end the chat.")
while True:
    user_message = input("You: ")
    if user_message.lower() == 'exit':
        print("Bot: Goodbye! Keep pushing forward!")
        break
    bot_reply = generate_response(user_message)
    print(f"Bot: {bot_reply}")


# In[ ]:




