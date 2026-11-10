
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# ==========================================
# 1. SAMPLE PRODUCT REVIEWS
# ==========================================

reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality."
]

# Assign IDs and topics
topics = [
    "Insulation", "Portability", "Lid", "Leakage",
    "Lid", "Cleaning", "Durability", "General",
    "Insulation", "Cleaning", "Capacity", "Quality",
    "Durability", "Delivery", "Insulation", "Price",
    "Comfort", "Quality"
]

df = pd.DataFrame({
    "id": [f"R{i:02}" for i in range(1, 19)],
    "text": reviews,
    "topic": topics
})

# ==========================================
# 2. SENTIMENT ANALYSIS USING VADER
# ==========================================

analyzer = SentimentIntensityAnalyzer()

def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score

df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text))
)

# ==========================================
# 3. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Lab 8 | Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("Product Feedback: Sentiment Analysis")

st.caption(
    "Data visualization of 18 synthetic product reviews. "
    "Automated sentiment labels should be reviewed by humans."
)

# ==========================================
# 4. SUMMARY METRICS
# ==========================================

col1, col2, col3 = st.columns(3)

col1.metric("Total Comments", len(df))
col2.metric("Positive Comments", (df["sentiment"] == "Positive").sum())
col3.metric("Negative Comments", (df["sentiment"] == "Negative").sum())

st.divider()

# ==========================================
# 5. SENTIMENT COUNT CHART
# ==========================================

st.subheader("1. Comment Counts by Sentiment")

order = ["Positive", "Neutral", "Negative"]

counts = df["sentiment"].value_counts().reindex(
    order, fill_value=0
)

st.bar_chart(counts, horizontal=True)

# ==========================================
# 6. WORD CLOUD
# ==========================================

st.subheader("2. Frequently Used Words")

# Retain "bottle" in the word cloud.
# Keep useful descriptive terms such as "cold", "leaks", and "lid".
stop_words = set(STOPWORDS) | {"drinks", "drink"}

cloud = WordCloud(
    width=900,
    height=500,
    background_color="white",
    colormap="viridis",
    stopwords=stop_words,
    collocations=False
).generate(" ".join(df["text"]))

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(cloud, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig, clear_figure=True)

st.caption(
    "Larger words appear more frequently in the comments. "
    "Word size represents frequency, not importance, satisfaction, "
    "or sentiment. The word 'bottle' is retained to show the "
    "product being discussed."
)

# ==========================================
# 7. ORIGINAL COMMENTS
# ==========================================

st.subheader("3. Review the Original Comments")

st.dataframe(
    df[["id", "text", "topic", "sentiment", "compound"]],
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Compound scores range from -1 to +1. "
    "They are sentiment estimates, not confidence ratings."
)

# ==========================================
# 8. MANUAL SENTIMENT AUDIT
# ==========================================

st.divider()
st.subheader("4. Manual Sentiment Audit")

st.write(
    "Review two comments from each sentiment category. "
    "Decide whether the automated label is reasonable."
)

audit = (
    df.groupby("sentiment", group_keys=False)
    .head(2)
    .copy()
)

audit["Your assessment"] = "Not reviewed"

edited_audit = st.data_editor(
    audit[["id", "text", "sentiment", "compound", "Your assessment"]],
    column_config={
        "Your assessment": st.column_config.SelectboxColumn(
            "Your assessment",
            options=["Not reviewed", "Reasonable", "Borderline", "Incorrect"],
            required=True
        )
    },
    disabled=["id", "text", "sentiment", "compound"],
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Compare your assessment with VADER's predicted label. "
    "Record why any borderline or incorrect classifications occurred."
)

# ==========================================
# 9. NEGATION TEST
# ==========================================

st.divider()
st.subheader("5. Testing Negation")

st.write(
    "Test how VADER responds when a positive statement is changed "
    "into a negative statement."
)

test_text = st.text_input(
    "Enter a product comment:",
    value="The bottle is not easy to clean."
)

if test_text.strip():
    test_sentiment, test_score = label_sentiment(test_text)

    st.write("Predicted sentiment:", test_sentiment)
    st.write("Compound score:", round(test_score, 4))

# ==========================================
# 10. SENTIMENT BY TOPIC
# ==========================================

st.divider()
st.subheader("6. Sentiment Comparison by Topic")

topic_counts = pd.crosstab(
    df["topic"],
    df["sentiment"]
).reindex(columns=order, fill_value=0)

st.bar_chart(topic_counts)

st.write(
    "This chart compares the number of positive, neutral, "
    "and negative comments across product topics."
)

st.caption(
    "The dataset is small and synthetic. Topic comparisons "
    "should be treated as exploratory rather than conclusive."
)

# ==========================================
# 11. INTERPRETATION
# ==========================================

st.divider()
st.subheader("7. Important Limitations")

st.markdown("""
- **Small dataset:** Only 18 comments are included.
- **Synthetic data:** The comments are examples rather than real customer feedback.
- **Context:** VADER may misunderstand sarcasm, mixed opinions, or negation.
- **Language:** VADER is designed primarily for English text.
- **Human validation:** Automated sentiment labels should be checked against the original comments.
""")

st.success("Lab 8 sentiment analysis dashboard completed!")
