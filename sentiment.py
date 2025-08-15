from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

NEGATIVE_WORDS = {"frustrated","upset","angry","annoyed","helpless","confused","overwhelmed","anxious","worried","stressed"}
POSITIVE_WORDS = {"thanks","grateful","confident","happy","relieved","great","awesome","helpful"}

def analyze_sentiment_and_mood(text: str):
    scores = analyzer.polarity_scores(text or "")
    comp = scores.get("compound", 0.0)
    if comp >= 0.05:
        label = "positive"
    elif comp <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    toks = set((text or "").lower().split())
    if label == "negative" and (toks & NEGATIVE_WORDS):
        mood = "frustrated"
    elif label == "positive" and (toks & POSITIVE_WORDS):
        mood = "relieved"
    else:
        mood = {"positive":"encouraged", "neutral":"calm", "negative":"concerned"}[label]

    return {
        "label": label,
        "compound": comp,
        "mood": mood,
        "raw": scores
    }
