import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

data = np.load("data/processed_data_A1.npz")
X = data["X"]
y = data["y"]

clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
clf.fit(X, y)

GESTURE_LABELS = {
    0: "Fist",
    1: "Wrist Flexion",
    2: "Wrist Extension",
    3: "Hand Open",
    4: "Pinch Index",
    5: "Pinch Middle",
    6: "Pinch Ring",
    7: "Pinch Little",
    8: "Tripod Pinch",
    9: "Thumb Up",
    10: "Thumb Down",
    11: "Neutral"
}
 
st.set_page_config(page_title="EMG Gesture UI", layout="wide")

st.title("EMG Gesture Recognition Demo")
st.write("Click the button to generate a new random EMG prediction.")

# Placeholders for UI content
placeholder_prediction = st.empty()
placeholder_probabilities = st.empty()

def display_prediction(idx):
    sample = X[idx].reshape(1, -1)
    pred = clf.predict(sample)[0]
    probs = clf.predict_proba(sample)[0]

    gesture_name = GESTURE_LABELS[pred]

    # Main prediction box
    placeholder_prediction.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background-color:#f0f2f6;">
            <h2 style="text-align:center; color:#2c7be5;">Predicted Gesture:</h2>
            <h1 style="text-align:center; color:#2c7be5;">{gesture_name}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Probability bars
    with placeholder_probabilities.container():
        st.subheader("Classifier Confidence")
        for i, p in enumerate(probs):
            st.progress(p)
            st.write(f"{GESTURE_LABELS[i]} — **{p:.2f}**")

if st.button("🔄 Generate New Gesture"):
    idx = np.random.randint(0, len(X))
    display_prediction(idx)
else:
    # Default: show the first sample on launch
    display_prediction(0)
