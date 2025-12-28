# Mental--Health-chatbot
A large-scale Mental Health Support Chatbot built using Machine Learning and Natural Language Processing (NLP).
The chatbot classifies user input into multiple mental health–related intents and provides supportive, empathetic responses.
It is deployed online using Streamlit Community Cloud.

# Live App: https://damini022-mental--health-chatbot-application-yywgss.streamlit.app

# Features
1) ML-based intent classification using TF-IDF and Logistic Regression
2) Hybrid response system (machine learning + template-based responses)
3) Covers multiple mental health conditions such as stress, anxiety, sadness, loneliness, anger, sleep issues, etc.
4) Crisis handling with safe fallback responses
5)Confidence-based fallback when prediction certainty is low
6) Fully deployed online (24×7 available)


# Tech Stack
1) Programming Language: Python
2)Libraries: scikit-learn ,numpy ,streamlit
3) Machine Learning Model: Logistic Regression
4)Vectorization: TF-IDF
5)Deployment: Streamlit Community Cloud
6)Version Control: GitHub


# How It Works
1) User enters a message describing their feelings
2) Text is vectorized using TF-IDF
3) A Logistic Regression model predicts the intent
4) Based on confidence score:
     High confidence → Intent-based response
     Low confidence → Clarification prompt
5) For crisis-related intents, the chatbot suggests seeking professional help

# Ethical Disclaimer
This chatbot is built only for educational purposes.
It does not replace professional mental health care.
If you or someone you know is experiencing severe distress, please contact a licensed mental health professional or a local helpline.
