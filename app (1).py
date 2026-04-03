
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("Financial Sentiment Analyzer")
st.write("Analyze sentiment of financial text using a fine-tuned model")

# ✅ Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load model
@st.cache_resource
def load_model():

    model_name = "ProsusAI/finbert"  # replace later if you saved your fine-tuned model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, tokenizer

model, tokenizer = load_model()

# ✅ Input
text = st.text_area("Enter financial text")

# ✅ Prediction
if st.button("Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter text.")

    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()

        labels = ["negative", "neutral", "positive"]
        sentiment = labels[pred]
        confidence = probs[0][pred].item()

        st.success("Analysis complete!")

        st.subheader("Predicted Sentiment")
        st.write(sentiment.upper())

        st.subheader("Confidence")
        st.write(f"{confidence:.2f}")
