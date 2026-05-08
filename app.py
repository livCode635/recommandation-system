import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Système de recommandation Item-Item",
    layout="wide"
)

st.title("Système de recommandation de films")
st.write("Collaborative Filtering item-item avec le dataset MovieLens.")

# =========================
# Chargement des données
# =========================

@st.cache_data
def load_data():
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv")

    df = ratings.merge(movies, on="movieId")

    # On garde les films les plus notés pour éviter une matrice trop lourde
    popular_movies = (
        df["movieId"]
        .value_counts()
        .head(300)
        .index
    )

    df = df[df["movieId"].isin(popular_movies)]

    return df, movies


df, movies = load_data()

st.sidebar.header("Paramètres")

top_n = st.sidebar.slider(
    "Nombre de recommandations",
    min_value=3,
    max_value=20,
    value=5
)

number_of_movies_to_rate = st.sidebar.slider(
    "Nombre de films proposés à noter",
    min_value=5,
    max_value=30,
    value=12
)

# =========================
# Films à noter
# =========================

st.subheader("Veuillez noter quelques films")

st.write(
    "Mettez une note aux films que vous connaissez. "
    "Laissez le à 0 si vous ne voulez pas noter un film."
)

movies_to_rate = (
    df[["movieId", "title", "genres"]]
    .drop_duplicates()
    .sample(number_of_movies_to_rate, random_state=42)
)

user_ratings = {}

cols = st.columns(2)

for index, row in enumerate(movies_to_rate.itertuples()):
    with cols[index % 2]:
        rating = st.slider(
            label=f"{row.title}",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.5,
            help=f"Genres : {row.genres}"
        )

        if rating > 0:
            user_ratings[row.movieId] = rating

# =========================
# Préparation de la matrice
# =========================

@st.cache_data
def build_similarity_matrix(df):
    user_item_matrix = df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    item_user_matrix = user_item_matrix.T

    similarity_matrix = cosine_similarity(item_user_matrix)

    item_similarity_df = pd.DataFrame(
        similarity_matrix,
        index=item_user_matrix.index,
        columns=item_user_matrix.index
    )

    return user_item_matrix, item_similarity_df


user_item_matrix, item_similarity_df = build_similarity_matrix(df)

# =========================
# Fonction de recommandation
# =========================

def recommend_movies(user_ratings, top_n=5):
    recommendations = {}

    rated_movie_ids = list(user_ratings.keys())
    all_movie_ids = item_similarity_df.index.tolist()

    unrated_movie_ids = [
        movie_id for movie_id in all_movie_ids
        if movie_id not in rated_movie_ids
    ]

    for movie_id in unrated_movie_ids:
        score = 0
        similarity_sum = 0

        for rated_movie_id, rating in user_ratings.items():
            if rated_movie_id in item_similarity_df.index:
                similarity = item_similarity_df.loc[movie_id, rated_movie_id]

                score += similarity * rating
                similarity_sum += abs(similarity)

        if similarity_sum > 0:
            recommendations[movie_id] = score / similarity_sum

    recommendations = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:top_n]

# =========================
# Recommandation
# =========================

st.divider()

st.subheader("Recommandations personnalisées")

if st.button("Générer mes recommandations", type="primary"):
    if len(user_ratings) == 0:
        st.warning("Veuillez noter au moins un film avant de générer les recommandations.")
    else:
        results = recommend_movies(user_ratings, top_n)

        if results:
            result_df = pd.DataFrame(results, columns=["movieId", "score"])

            result_df = result_df.merge(
                movies[["movieId", "title", "genres"]],
                on="movieId",
                how="left"
            )

            result_df = result_df[["title", "genres", "score"]]
            result_df["score"] = result_df["score"].round(2)

            st.success(f"Voici vos {top_n} meilleures recommandations")
            st.dataframe(result_df, use_container_width=True)

            for row in result_df.itertuples():
                st.markdown(
                    f"""
                    <div style="
                        padding: 18px;
                        margin-bottom: 12px;
                        border-radius: 18px;
                        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(16, 185, 129, 0.12));
                        border: 1px solid rgba(14, 165, 233, 0.25);
                    ">
                        <h4 style="margin-bottom: 6px;">{row.title}</h4>
                        <p style="margin: 0; color: #64748b;">Genres : {row.genres}</p>
                        <p style="margin-top: 8px;"><strong>Score :</strong> {row.score}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.error("Aucune recommandation trouvée.")

# =========================
# Debug léger
# =========================

with st.expander("Voir un aperçu des matrices"):
    st.write("Données MovieLens")
    st.dataframe(df.head(20), use_container_width=True)

    st.write("Matrice utilisateur-film")
    st.dataframe(user_item_matrix.iloc[:10, :10], use_container_width=True)

    st.write("Matrice de similarité item-item")
    st.dataframe(item_similarity_df.iloc[:10, :10], use_container_width=True)