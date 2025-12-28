import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random

# -----------------------------
# LARGE-SCALE DATASET (HYBRID)
# -----------------------------
data = {
    "intents": [
        {
            "tag": "sadness",
            "patterns": [
                "I feel sad", "Feeling low", "I feel empty",
                "Nothing feels right", "I want to cry",
                "I feel emotionally drained", "I feel hopeless",
                "I feel broken", "I am not okay"
            ]
        },
        {
            "tag": "anxiety",
            "patterns": [
                "I feel anxious", "I'm worried",
                "Feeling nervous", "My heart is racing",
                "I feel uneasy", "I can't calm down",
                "I feel scared", "Panic feeling"
            ]
        },
        {
            "tag": "stress",
            "patterns": [
                "I'm stressed", "Too much pressure",
                "Overwhelmed with work", "College stress",
                "I can't handle this", "I feel burnt out",
                "Workload is too much"
            ]
        },
        {
            "tag": "loneliness",
            "patterns": [
                "I feel lonely", "I feel alone",
                "Nobody understands me",
                "I have no one to talk to",
                "I feel isolated"
            ]
        },
        {
            "tag": "motivation_loss",
            "patterns": [
                "I feel unmotivated", "No motivation",
                "I don't feel like doing anything",
                "I feel lazy all the time",
                "I feel stuck in life"
            ]
        },
        {
            "tag": "sleep_problem",
            "patterns": [
                "I can't sleep", "I have insomnia",
                "Poor sleep", "I wake up at night",
                "Sleep issues"
            ]
        },
        {
            "tag": "anger",
            "patterns": [
                "I'm angry", "I feel irritated",
                "I'm frustrated", "I want to scream",
                "I feel rage"
            ]
        },
        {
            "tag": "crisis",
            "patterns": [
                "I want to hurt myself",
                "I feel like ending everything",
                "I don't want to live",
                "I feel suicidal"
            ]
        }
    ]
}

# -----------------------------
# RESPONSE TEMPLATES
# -----------------------------
response_templates = {
    "sadness": [
        "I'm really sorry you're feeling this way.",
        "It sounds like you're going through a tough time.",
        "Thank you for opening up about how you're feeling."
    ],
    "anxiety": [
        "Anxiety can feel overwhelming, but you're safe right now.",
        "Try taking slow, deep breaths.",
        "It's okay to feel anxious sometimes."
    ],
    "stress": [
        "Stress can build up quickly — let's slow things down.",
        "Try focusing on one small thing at a time.",
        "You're doing your best, and that matters."
    ],
    "loneliness": [
        "You're not alone right now — I'm here with you.",
        "It can really hurt to feel lonely.",
        "I'm glad you reached out."
    ],
    "motivation_loss": [
        "It's okay to feel unmotivated sometimes.",
        "Try starting with one very small task.",
        "Progress doesn't have to be perfect."
    ],
    "sleep_problem": [
        "Sleep problems can affect mental health a lot.",
        "Try keeping a consistent sleep schedule.",
        "Avoid screens before bedtime if possible."
    ],
    "anger": [
        "It's okay to feel angry — let's try to calm your body first.",
        "Take a deep breath and pause for a moment.",
        "Strong emotions can pass with time."
    ]
}

# -----------------------------
# PREPARE TRAINING DATA
# -----------------------------
sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(sentences)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# -----------------------------
# CHATBOT LOGIC
# -----------------------------
def chatbot_response(user_input):
    vec = vectorizer.transform([user_input])
    intent = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()

    # Crisis handling
    if intent == "crisis":
        return (
            "I'm really glad you reached out. "
            "You deserve support. Please consider contacting a mental health professional "
            "or a local helpline immediately."
        )

    # Low confidence fallback
    if confidence < 0.45:
        return "I'm not fully sure I understood you. Can you tell me more?"

    return random.choice(response_templates.get(
        intent,
        ["I'm here to listen. Tell me more."]
    ))

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🧠 Mental Health Support Chatbot")
st.write(
    "⚠️ This chatbot is for educational purposes only and does not replace professional mental health care."
)

user_input = st.text_input("How are you feeling today?")

if user_input:
    st.write("**Bot:**", chatbot_response(user_input))
