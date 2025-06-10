import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

st.title("Customer Support Ticket Analysis Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        if pd.isnull(text):
            return []
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return filtered_tokens

    df['tokens'] = df['Ticket Description'].apply(preprocess_text)

    all_tokens = [token for tokens in df['tokens'] for token in tokens]
    counter = Counter(all_tokens)
    most_common_words = counter.most_common(20)

    st.subheader("Top 20 Most Common Words")
    st.table(most_common_words)

    # Plot
    words, freqs = zip(*most_common_words)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(words, freqs)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Date conversions
    df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
    df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], errors='coerce')
    df['Resolution Duration (hours)'] = (df['Time to Resolution'] - df['Date of Purchase']).dt.total_seconds() / 3600

    if 'Ticket Type' in df.columns:
        avg_resolution_by_type = df.groupby('Ticket Type')['Resolution Duration (hours)'].mean().sort_values()
        st.subheader("Average Resolution Time by Ticket Type (hours)")
        st.bar_chart(avg_resolution_by_type)

    top_slow_resolutions = df.sort_values(by='Resolution Duration (hours)', ascending=False).head(5)
    st.subheader("Top 5 Tickets with Longest Resolution Time")
    st.dataframe(top_slow_resolutions[['Ticket ID', 'Ticket Description', 'Ticket Type', 'Resolution Duration (hours)']])

    def generate_recommendations():
        recommendations = []
        if most_common_words:
            recommendations.append(
                f"1. Automate responses or create FAQ for frequent issues like '{most_common_words[0][0]}' to improve handling time."
            )
        if 'Ticket Type' in df.columns:
            slow_types = avg_resolution_by_type.tail(2).index.tolist()
            recommendations.append(
                f"2. Provide targeted training or resources for ticket types with higher resolution times: {', '.join(slow_types)}."
            )
        recommendations.append("3. Enhance self-service portals to cover top recurring issues.")
        recommendations.append("4. Optimize ticket routing for faster resolution of complex tickets.")
        recommendations.append("5. Continuously monitor ticket data to adapt processes proactively.")
        return recommendations

    recommendations = generate_recommendations()

    st.subheader("Recommendations")
    for rec in recommendations:
        st.markdown(f"- {rec}")

    with open("summary_report.txt", "w") as file:
        file.write("--- Customer Support Ticket Analysis Report ---\n")
        file.write(f"Total Tickets Analyzed: {len(df)}\n\n")
        file.write("Top Frequent Issues (Words):\n")
        for word, freq in most_common_words[:10]:
            file.write(f"- {word} ({freq} times)\n")
        if 'Ticket Type' in df.columns:
            file.write("\nAverage Resolution Time by Ticket Type:\n")
            file.write(avg_resolution_by_type.to_string())
            file.write("\n")
        file.write("\nRecommendations:\n")
        for rec in recommendations:
            file.write(f"- {rec}\n")

    st.success("Summary report saved to 'summary_report.txt'")
