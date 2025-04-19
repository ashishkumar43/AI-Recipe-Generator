import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Indian_Food_Dataset.csv")
    df = df.dropna(subset=["TranslatedIngredients", "TranslatedInstructions"])
    df["clean_ingredients"] = df["TranslatedIngredients"].apply(preprocess)
    return df

# Preprocess ingredients
def preprocess(text):
    doc = nlp(str(text).lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Recommend recipes
def get_recipes(user_ingredients, df, vectorizer, ingredient_vectors, top_n=5):
    cleaned_input = preprocess(user_ingredients)
    user_vector = vectorizer.transform([cleaned_input])
    similarity_scores = cosine_similarity(user_vector, ingredient_vectors).flatten()
    top_matches = similarity_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_matches]

# Streamlit 
st.title("🍲 AI Recipe Generator")
st.write("Enter the ingredients you have, and get recipe suggestions!")

# User input
user_input = st.text_input("Enter ingredients (comma separated):")

# Load data and prepare vectorizer
df = load_data()
vectorizer = TfidfVectorizer()
ingredient_vectors = vectorizer.fit_transform(df["clean_ingredients"])

if user_input:
    results = get_recipes(user_input, df, vectorizer, ingredient_vectors)
    for i, row in results.iterrows():
        st.markdown("---")
        st.subheader(f"🍽️ {row['TranslatedRecipeName']}")
        st.markdown(f"**🥘 Cuisine**: {row['Cuisine'] if 'Cuisine' in row else 'Unknown'}")
        st.markdown(f"**🧂 Ingredients**: {row['TranslatedIngredients']}")
        st.markdown(f"**📖 Instructions**: {row['TranslatedInstructions']}")
