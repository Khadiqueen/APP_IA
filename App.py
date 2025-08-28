# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pickle
import requests
import uuid
import os
from gtts import gTTS
from groq import Groq
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # charge .env dans os.environ

# --- Feedback (likes/dislikes) partagé entre l'app et le notebook ---
from pathlib import Path
import json, time, os

FEEDBACK_PATH = Path("data") / "chat_history.json"
FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)  # crée ./data/ si absent

def log_feedback(mid: int, action: str, path: str | os.PathLike = FEEDBACK_PATH):
    """
    mid: TMDB movie_id
    action: "like" ou "dislike"
    """
    entry = {"mid": int(mid), "action": str(action), "ts": time.time()}
    try:
        data = json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else []
    except Exception:
        data = []
    data.append(entry)
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

from pathlib import Path
import json, time, os

BASE_DIR = Path(__file__).resolve().parent       # ← dossier de App.py (APP/)
FEEDBACK_PATH = BASE_DIR / "data" / "chat_history.json"
FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_feedback(mid: int, action: str, path: Path = FEEDBACK_PATH):
    entry = {"mid": int(mid), "action": str(action), "ts": time.time()}
    try:
        data = json.load(open(path, "r", encoding="utf-8")) if path.exists() else []
    except Exception:
        data = []
    data.append(entry)
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)



# ================== CONFIG GLOBALE ==================
st.set_page_config(page_title="FilmScope IA", page_icon="🎥", layout="wide")

# ======== Invalidation AUTOMATIQUE du cache (version + fichiers + env) ========
APP_VERSION = "1.0.0"  # incrémente quand tu veux forcer un refresh global

def _compute_cache_fingerprint() -> str:
    parts = [APP_VERSION, os.getenv("TMDB_API_KEY", "")]
    for fp in ("movie_list.pkl", "similarity.pkl", ".env"):
        try:
            parts.append(str(int(os.path.getmtime(fp))))
        except Exception:
            parts.append("0")
    return "|".join(parts)

_FP = _compute_cache_fingerprint()
if st.session_state.get("_CACHE_FP") != _FP:
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    st.session_state["_CACHE_FP"] = _FP
# ========================================================================
# ========================================================================
# --------- Thème : Noir / Doré / Bordeaux (look Netflix/Amazon) ---------
st.markdown("""
<style>
:root {
  --noir:#000;
  --blanc:#fff;
  --dore:#FFD700;
  --bordeaux:#7A1F1F;
  --fond:#0b0b0b;
  --fond-side:#111;
  --gris:#1a1a1a;

  /* Unification des tailles */
  --btn-h: 40px;
  --btn-radius: 12px;
  --poster-w: 220px;
  --poster-h: 330px;
}

/* Base */
html, body, [class*="css"] {
  background: var(--noir) !important;
  color: var(--blanc) !important;
  font-family: "Times New Roman", Times, serif !important;
  font-weight: 400 !important;     /* normal (pas gras) */
  font-size: 1.05rem !important;   /* +5% environ */
}

/* Boutons (taille réduite & uniforme) */
.stButton>button {
  background: linear-gradient(90deg, var(--dore), var(--bordeaux));
  color: var(--noir) !important;
  font-weight: 800;
  border-radius: var(--btn-radius);
  height: var(--btn-h);
  padding: 0 14px;
  font-size: .95rem;
  width: auto;
  min-width: 140px;
  transition: transform .12s ease-in-out, box-shadow .12s ease-in-out;
  box-shadow: 0 2px 10px rgba(122,31,31,0.35);
  border: 1px solid rgba(255,215,0,0.35);
}
.stButton>button:hover {
  transform: translateY(-1px) scale(1.01);
  box-shadow: 0 4px 14px rgba(255,215,0,0.25);
}

/* Cartes */
.card {
  border: 1px solid rgba(255,215,0,0.25);
  border-radius: 16px;
  padding: 10px;
  background: var(--gris);
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
  height: 100%;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(122,31,31,0.25);
  border-color: rgba(255,215,0,0.45);
}
.card-title { font-weight: 800; margin: 6px 0 4px 0; }
.card-meta { color: rgba(255,255,255,0.85); font-size: 13px; margin-bottom: 6px; }
.card-overview {
  color: rgba(255,255,255,0.92);
  font-size: 14px;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(255,215,0,0.15);
  color: var(--dore);
  font-size: 12px;
  margin-right: 6px;
}

/* Titres */
.main-title {
  text-align: center;
  font-size: 46px;
  color: var(--blanc);
  margin-bottom: 6px;
  font-weight: 900;
  letter-spacing: .5px;
}
.subtitle {
  text-align: center;
  font-size: 18px;
  color: rgba(255,255,255,0.9);
  margin-bottom: 18px;
}
.hr {
  border: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--dore), var(--bordeaux), transparent);
  margin: 12px 0 24px 0;
}

/* HERO */
.hero {
  background: linear-gradient(180deg, rgba(0,0,0,0.65), rgba(0,0,0,0.85)),
              url('https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?q=80&w=1600&auto=format&fit=crop')
              center/cover no-repeat;
  border-radius: 18px;
  padding: 38px 24px;
  margin: 10px 0 18px 0;
  border: 1px solid rgba(255,215,0,0.25);
}
.hero h1 { font-size: 34px; margin: 0 0 6px 0; color: var(--dore); }
.hero p { color: rgba(255,255,255,0.92); margin: 0 0 10px 0; }

/* Footer */
.footer-global {
  margin-top: 28px;
  padding: 12px 0;
  text-align: center;
  font-size: 13.5px;
  color: rgba(255,255,255,0.9);
  border-top: 1px solid rgba(255,215,0,0.28);
}
.footer-global a { color: var(--dore); text-decoration: none; }

/* Images: taille UNIFORME + hover — uniquement dans le contenu principal */
[data-testid="stAppViewContainer"] .stImage img,
[data-testid="stAppViewContainer"] .poster-fixed {
  width: var(--poster-w) !important;
  height: var(--poster-h) !important;
  object-fit: cover !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,215,0,0.25);
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
[data-testid="stAppViewContainer"] .stImage img:hover,
[data-testid="stAppViewContainer"] .poster-fixed:hover {
  transform: translateY(-2px) scale(1.02);
  box-shadow: 0 8px 24px rgba(255,215,0,0.18);
  border-color: rgba(255,215,0,0.45);
}

/* Vignette avec badge circulaire (genres) */
.poster-wrap {
  position: relative;
  display: inline-block;
  width: var(--poster-w);
  height: var(--poster-h);
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(255,215,0,0.25);
  background: #000;
}
.poster-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.genre-badge-circle {
  position: absolute;
  top: 10px;
  left: 10px;
  width: 50px;
  height: 50px;
  min-width: 50px;
  border-radius: 50%;
  background: rgba(255,215,0,0.92);
  color: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 900;
  font-size: 11px;
  text-align: center;
  line-height: 1.05;
  padding: 6px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.35);
  border: 1px solid rgba(122,31,31,0.35);
}
.genre-badge-circle small {
  display: block;
  font-size: 10px;
  font-weight: 800;
}

/* Mini ajustement des selects pour compacité */
.stSelectbox label, .stMultiSelect label {
  font-size: .95rem;
}
</style>
""", unsafe_allow_html=True)


# --- Override CSS pour le logo du sidebar (plein, sans hover ni cadre)
st.markdown("""
<style>
/* Sidebar : logo plat, sans cadre ni hover */
[data-testid="stSidebar"] .stImage,
[data-testid="stSidebar"] .stImage > figure{
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* Image du logo : pleine largeur, proportions respectées, aucun effet */
[data-testid="stSidebar"] .stImage img{
  width: 100% !important;
  max-width: 520px !important;
  height: auto !important;
  object-fit: contain !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  transform: none !important;
  transition: none !important;
}

/* Neutralise tout hover */
[data-testid="stSidebar"] .stImage img:hover{
  transform: none !important;
  box-shadow: none !important;
  border: none !important;
}
</style>
""", unsafe_allow_html=True)



# ================== IMPORTS OPTIONNELS ==================
try:
    from streamlit_player import st_player
except Exception:
    st_player = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

# ---------- Helpers de cache pour TMDB (à mettre au-dessus des pages) ----------
@st.cache_data(show_spinner=False, ttl=3600)
def tmdb_movie_cached(mid: int):
    return tmdb_get(f"movie/{mid}") or {}

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_overview_vote_genres_fast(movie_id):
    data = tmdb_movie_cached(movie_id)
    overview = data.get("overview") or "Description indisponible..."
    genres = ", ".join([g["name"] for g in data.get("genres", [])]) or "Genres indisponibles"
    return overview, genres, data

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster_fast(movie_id, size="w185"):
    data = tmdb_movie_cached(movie_id)
    p = data.get('poster_path')
    return f"https://image.tmdb.org/t/p/{size}{p}" if p else "https://via.placeholder.com/300x450.png?text=No+Image"

# ================== DONNÉES ==================
def load_pickles():
    try:
        movies = pickle.load(open('movie_list.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        if 'title' not in movies.columns or 'movie_id' not in movies.columns:
            raise ValueError("Colonnes attendues absentes (title, movie_id)")
        return movies, similarity
    except Exception as e:
        st.warning(f"⚠️ Données de reco indisponibles ({e}). Fonctions limitées.")
        return pd.DataFrame(columns=['title','movie_id']), None

movies, similarity = load_pickles()

# ================== TMDB ==================
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8")
TMDB_LANG = "fr-FR"

def tmdb_get(path, params=None):
    base = "https://api.themoviedb.org/3"
    params = params or {}
    params["api_key"] = TMDB_API_KEY
    params["language"] = TMDB_LANG
    try:
        r = requests.get(f"{base}/{path}", params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def fetch_poster(movie_id, size="w300"):
    data = tmdb_get(f"movie/{movie_id}")
    p = data.get('poster_path')
    return f"https://image.tmdb.org/t/p/{size}{p}" if p else "https://via.placeholder.com/300x450.png?text=No+Image"

def fetch_overview_vote_genres(movie_id):
    data = tmdb_get(f"movie/{movie_id}")
    overview = data.get("overview") or "Description indisponible."
    vote = data.get("vote_average", 0.0)
    genres = ", ".join([g["name"] for g in data.get("genres", [])]) or "Genres indisponibles"
    return overview, vote, genres, data

def get_trailer_url(movie_id):
    data = tmdb_get(f"movie/{movie_id}/videos")
    for v in data.get("results", []):
        if v.get("type") == "Trailer" and v.get("site") == "YouTube":
            return f"https://www.youtube.com/watch?v={v['key']}"
    return None

def tmdb_similar(movie_id, limit=6):
    data = tmdb_get(f"movie/{movie_id}/similar")
    return [m["id"] for m in data.get("results", [])][:limit]

def tmdb_collection(movie_id):
    _, _, _, data = fetch_overview_vote_genres(movie_id)
    col = data.get("belongs_to_collection")
    if not col:
        return []
    col_id = col.get("id")
    c = tmdb_get(f"collection/{col_id}") if col_id else {}
    return [m["id"] for m in c.get("parts", [])] if c else []

def tmdb_popular(limit=12, reverse=False):
    p1 = tmdb_get("movie/popular", params={"page": 1}).get("results", [])
    p2 = tmdb_get("movie/popular", params={"page": 2}).get("results", [])
    allm = p1 + p2
    allm = sorted(allm, key=lambda m: m.get("popularity", 0), reverse=not reverse)
    return allm[:limit]

def tmdb_by_genre(genre_ids, limit=12):
    res = tmdb_get("discover/movie", params={"with_genres": ",".join(map(str, genre_ids)), "sort_by": "vote_average.desc", "vote_count.gte": 50})
    return res.get("results", [])[:limit]

def get_movie_id(title):
    row = movies[movies['title'].str.lower() == title.lower()]
    return int(row.iloc[0]['movie_id']) if not row.empty else None

def tmdb_search_title(title: str):
    js = tmdb_get("search/movie", params={"query": title, "include_adult": False})
    return js.get("results", [])

def tmdb_details_full(movie_id: int):
    d = tmdb_get(f"movie/{movie_id}")
    release = d.get("release_date") or ""
    runtime = d.get("runtime") or 0
    genres = ", ".join([g["name"] for g in d.get("genres", [])]) or "Genres indisponibles"
    vote = float(d.get("vote_average") or 0)
    vc = int(d.get("vote_count") or 0)
    pop = d.get("popularity", 0)
    poster = f"https://image.tmdb.org/t/p/w500{d.get('poster_path')}" if d.get("poster_path") else "https://via.placeholder.com/300x450.png?text=No+Image"
    overview = d.get("overview") or "Description indisponible."
    title = d.get("title") or d.get("name") or "Titre indisponible"
    return {
        "title": title, "overview": overview, "vote": vote, "vote_count": vc,
        "popularity": pop, "genres": genres, "release_date": release,
        "runtime": runtime, "poster": poster
    }

# ================== UTILS RECO (nécessaires aux pages) ==================
def recommend_titles_by_title(title, k=6):
    if movies.empty or title not in movies['title'].values or similarity is None:
        return []
    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:k+1]
    return [movies.iloc[i].title for i, _ in scores]

def recommend_by_genres(genres, limit=8):
    if movies.empty:
        return []
    titles = []
    for _, row in movies.sample(min(200, len(movies))).iterrows():
        mid = int(row['movie_id'])
        _, _, g, _ = fetch_overview_vote_genres(mid)
        if any(gg.strip().lower() in g.lower() for gg in genres):
            titles.append(row['title'])
        if len(titles) >= limit:
            break
    return list(dict.fromkeys(titles))[:limit]



# ================== SIDEBAR ==================
logo_path = "images/film.png" if os.path.exists("images/film.png") else None

with st.sidebar:
    if logo_path:
        st.markdown('<div class="sidebar-logo" style="text-align:center; padding:0; margin:0;">', unsafe_allow_html=True)
        st.image(logo_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Navigation (unique)
    page = st.selectbox(
        "Navigation",
        ["🏠 Accueil","📊 Visualisation" ,"🎬 Recommandation", "📈 Prédiction Machine Learning",
         "🧠 Recommandation avancée", "💬 FilmScope Chatbot"],
        key="nav_page_select"
    )
 
    # Contenu spécifique au Chatbot dans le même sidebar (historique + bouton clear)
    if page == "💬 FilmScope Chatbot":
        st.markdown("---")
        st.header("🗂️ Historique")
        if st.session_state.get("history"):
            for m in reversed(st.session_state.history):
                who = "👤" if m["role"] == "user" else "🎬"
                st.write(f"{who} {m['content']}")
        else:
            st.caption("L'historique apparaît ici après vos échanges.")

        if st.button("🧹 Effacer l'historique", key="clear_hist_btn", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_audio_b64 = None
            st.toast("Historique effacé.")

# === Historique: sauvegarde/chargement ===
HIST_FILE = "chat_history.json"
def save_chat_history():
    try:
        with open(HIST_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Erreur sauvegarde historique : {e}")

def load_chat_history():
    if os.path.exists(HIST_FILE):
        try:
            with open(HIST_FILE, "r", encoding="utf-8") as f:
                st.session_state.chat_history = json.load(f)
        except Exception:
            st.session_state.chat_history = []



#--------- PARTIE 1---------

# ================== PAGES ==================
if page == "🏠 Accueil":
    # ====== TITRE EN HAUT ======
    st.markdown('<div class="main-title">🎥 Bienvenue sur FilmScope IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sélectionnez vos genres et laissez la magie opérer.</div>', unsafe_allow_html=True)
    
    # ====== HERO EN BAS (déplacé ici) ======
    st.markdown(f"""
    <div class="hero" style="
        position: relative;
        border-radius: 18px;
        padding: 38px 24px;
        margin: 10px 0 18px 0;
        overflow: hidden;
        border: 1px solid rgba(230,196,110,0.25);
    ">
      <div style="
          background: url('images/persone.jpg') center/cover no-repeat;
          filter: blur(6px);
          position: absolute; top:0; left:0; right:0; bottom:0; z-index:0;
      "></div>
      <div style="
          background: linear-gradient(180deg, rgba(230,196,110,0.45), rgba(0,0,0,0.88));
          position: absolute; top:0; left:0; right:0; bottom:0; z-index:1;
      "></div>
      <div style="position: relative; z-index: 2;">
        <h1 style="color:#E6C46E; margin:0;">FilmScope IA</h1>
        <p style="color: rgba(246,242,233,0.95); margin:6px 0 0 0; font-size:16.5px;">
          <b>Explorez, analysez, prédisez</b> — Parlez d'un <b>genre</b>,
          et on vous sert le film parfait. <i>Moins de scroll, plus d’émotions.</i>
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 C’est quoi **FilmScope** ?")
    st.markdown("""
        FilmScope IA est une application innovante qui combine l’intelligence artificielle 
        et l’analyse cinématographique pour recommander des films parfaitement adaptés à vos envies. 
        Elle aide les passionnés et curieux à découvrir, explorer et apprécier le cinéma grâce à des suggestions personnalisées, 
        des résumés clairs et des visuels immersifs
    """)

    st.markdown("### 💡 Pourquoi **FilmScope IA** ?")
    st.markdown(""" **Parce-que** elle fournit  :
    -🎯 Précision – Des recommandations taillées sur mesure selon vos goûts et humeurs.
    -⚡ Rapidité – Trouvez en quelques secondes le film parfait, sans scroll interminable.
    -🌍 Ouverture – Un catalogue qui valorise autant le cinéma africain qu’international.
    -🗣️ Interaction – Un chatbot expert cinéma qui répond en français .
    -🎬 Immersion – Résumés clairs, visuels soignés et suggestions enrichissantes. 
    """)

    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

    # ===== Choix des genres + Suggestions =====
    GENRES = ["Action","Drama","Comedy","Romance","Science Fiction","Thriller","Horror","Animation"]
    picked_genres = st.multiselect(
        "🎭 Choisissez les genres que vous voulez regarder :",
        GENRES,
        default=["Action","Drama"],
        help="Vous pouvez en choisir plus de genre pour affiner la sélection."
    )

    # Styles locaux : badge au-dessus + image légèrement plus grande + hover centré, sans ombre
    st.markdown("""
    <style>
    .sugg-card{display:flex;flex-direction:column;align-items:center;gap:8px}
    .choice-pill{
      display:inline-flex;align-items:center;justify-content:center;
      padding:6px 12px;margin-bottom:8px;
      border-radius:999px;background:rgba(255,215,0,0.95);color:#000;
      font-weight:900;font-size:12px;line-height:1.1;border:1px solid rgba(122,31,31,0.30);
      max-width:var(--poster-w);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
    }
    .poster-wrap{
      position:relative;display:inline-block;
      width:var(--poster-w);height:var(--poster-h);
      border-radius:12px;overflow:hidden;
      border:1px solid rgba(255,215,0,0.25);background:#000;
      transition:border-color .15s ease;
      box-shadow:none;
    }
    .poster-wrap img{
      width:100%;height:100%;object-fit:cover;display:block;
      will-change:transform;backface-visibility:hidden;
      transition:transform .18s ease;
      transform:scale(1.04);
      transform-origin:center center;
    }
    .poster-wrap:hover img{ transform:scale(1.10); }
    .poster-wrap:hover{ border-color:rgba(255,215,0,0.45); }
    </style>
    """, unsafe_allow_html=True)

    def _badge_text_full(gs: list[str]) -> str:
        return "Vous avez choisi : " + ", ".join(gs) if gs else "Vous avez choisi : Tous genres"

    # 🔧 Normalisation FR/EN pour le matching local
    import unicodedata, re
    def _norm_txt(s: str) -> str:
        s = unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode("ascii")
        s = re.sub(r"[-–—]", " ", s)
        return s.lower().strip()

    # Mapping TMDB pour fallback robuste
    TMDB_GENRE_IDS = {
        "Action": 28, "Drama": 18, "Comedy": 35, "Romance": 10749,
        "Science Fiction": 878, "Thriller": 53, "Horror": 27, "Animation": 16
    }

    if st.button("✨ Suggestions pensées pour vous", key="btn_suggest_home"):
        items: list[tuple[str, int]] = []

        # 1) Moteur local standard
        try:
            local_titles = recommend_by_genres(picked_genres or ["Action"], limit=8) or []
            for t in local_titles:
                mid = get_movie_id(t)
                if mid:
                    items.append((t, mid))
        except Exception:
            local_titles = []

        # 1bis) Renfort local FR/EN si rien trouvé
        if not items and not movies.empty and picked_genres:
            try:
                for _, row in movies.sample(min(600, len(movies))).iterrows():
                    try:
                        mid = int(row["movie_id"])
                    except Exception:
                        continue
                    _, g, _data = fetch_overview_vote_genres_fast(mid)
                    if any(_norm_txt(gg) in _norm_txt(g) for gg in picked_genres):
                        items.append((row["title"], mid))
                        if len(items) >= 8:
                            break
            except Exception:
                pass

        # 2) Fallback TMDB si toujours rien
        if not items:
            genre_ids = [TMDB_GENRE_IDS[g] for g in picked_genres if g in TMDB_GENRE_IDS]
            try:
                tmdb_res = tmdb_by_genre(genre_ids, limit=8) if genre_ids else tmdb_popular(limit=8)
            except Exception:
                tmdb_res = []
            for m in tmdb_res:
                title = m.get("title") or m.get("name") or "Titre indisponible"
                mid = m.get("id")
                if mid:
                    items.append((title, mid))

        if not items:
            st.warning("Aucune suggestion trouvée. Essayez d’autres genres.")
        else:
            st.caption("🎬 Résultats pensés pour vous — *sélection courte et efficace*")
            cols = st.columns(4)
            badge_text = _badge_text_full(picked_genres)

            for i, (t, mid) in enumerate(items):
                poster = fetch_poster(mid) if mid else "https://via.placeholder.com/300x450.png?text=No+Image"
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class='card sugg-card' style="--poster-w:260px; --poster-h:390px;">
                      <div class="choice-pill">{badge_text}</div>
                      <div class="poster-wrap">
                        <img src="{poster}" alt="{t}">
                      </div>
                      <div style="text-align:center;font-weight:800;max-width:var(--poster-w)">{t}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

    # ===== Bande-annonce =====
    st.markdown("#### 🎬 Une bande-annonce tout de suite ?")
    st.caption("Choisissez un titre, on ouvre la meilleure bande-annonce dispo.")
    if not movies.empty:
        film_choice = st.selectbox("Choisissez un film :", movies['title'].values)
        if st.button("▶️ Lancer la bande-annonce", key="btn_launch_trailer"):
            mid = get_movie_id(film_choice)
            if mid:
                url = get_trailer_url(mid)
                if url and st_player:
                    st_player(url)
                elif url:
                    st.markdown(f"[Ouvrir la bande-annonce sur YouTube]({url})")
                else:
                    st.warning("Aucune bande-annonce officielle trouvée.")
    else:
        st.info("Aucune donnée de films disponible pour le moment.")

    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

    # ===== À propos =====
    st.markdown("### ✨ À propos")
    st.caption("« FilmScope IA — le cinéma, version assistée »")
    st.markdown("""
    **👩‍💻 Développée par :** MOHAMED KHADIJA  
    **📅 Date :** 30/08/2025 • **☎️ Contact :** +237 691203120  
    """)

    # Barrière anti-superposition
    st.stop()



# ---- PARTIE 2 ----------
# ------VISUALISATION-------
elif page == "📊 Visualisation":
    import io, pickle, unicodedata
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime, timedelta

    # === Palette & style unifié ===
    GOLD = "#FFD700"
    BORDEAUX = "#7A1F1F"
    NOIR = "#0b0b0b"
    BLANC = "#FFFFFF"

    COLOR_SEQ = [GOLD, BORDEAUX, "#caa24c", "#b55656"]  # doré -> bordeaux
    HEATMAP_SCALE = [(0, NOIR), (0.5, BORDEAUX), (1, GOLD)]

    # 🔄 Réinitialiser les caches à l'entrée de cette page
    if st.session_state.get("_last_page") != "📊 Visualisation":
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        st.session_state["_last_page"] = "📊 Visualisation"

    # ---------- Styles UI ----------
    st.markdown(
        """
<style>
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, var(--dore), var(--bordeaux));
    color: var(--noir) !important; font-weight: 800;
    border-radius: var(--btn-radius); height: 32px; padding: 0 10px;
    font-size: .85rem; min-width: 110px; transition: transform .12s, box-shadow .12s;
    box-shadow: 0 2px 8px rgba(122,31,31,0.28);
    border: 1px solid rgba(255,215,0,0.35);
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform: translateY(-1px) scale(1.01);
    box-shadow: 0 4px 12px rgba(255,215,0,0.22);
}
.kpi-strip { display:flex; flex-wrap:nowrap; gap:12px; margin:10px 0; padding:6px 2px;
    overflow-x:auto; overflow-y:hidden; -webkit-overflow-scrolling:touch; scrollbar-width:thin; }
.kpi-strip::-webkit-scrollbar { height:6px; } .kpi-strip::-webkit-scrollbar-track { background:transparent; }
.kpi-strip::-webkit-scrollbar-thumb { background: rgba(255,215,0,.35); border-radius:4px; }
.kpi-card { flex:0 0 150px; width:150px; height:150px; background: var(--gris);
    border:1px solid rgba(255,215,0,0.28); border-radius:16px; box-shadow:0 6px 18px rgba(0,0,0,.25);
    position:relative; display:flex; align-items:center; justify-content:center; transition: transform .12s, box-shadow .12s, border-color .12s; }
.kpi-card:hover { transform: translateY(-2px) scale(1.02); box-shadow:0 10px 24px rgba(255,215,0,0.18); border-color:rgba(255,215,0,0.45); }
.kpi-v { font-weight:900; font-size:32px; color:#FFD700; line-height:1; }
.kpi-l { position:absolute; left:8px; right:8px; bottom:10px; text-align:center; font-size:12px; color:#fff; opacity:.95; }
.tip-bubble { display:block; border:1px dashed rgba(255,215,0,.6); border-radius:10px; background:rgba(255,215,0,.06);
    padding:8px 10px; margin:6px 0 10px 0; color:#fff; font-size:.95rem; }
div[data-testid="stSelectbox"] > div { gap:4px; } div[data-testid="stSelectbox"] div[role="combobox"] { min-height:32px; }
div[data-testid="stSelectbox"] label p { font-size:12px; margin-bottom:2px; }
</style>
""",
        unsafe_allow_html=True,
    )

    # ---------- Thème Plotly ----------
    def _themed(fig):
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=BLANC),
            legend=dict(title_text="", font=dict(color=BLANC)),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig

    # ---------- TITRE ----------
    st.markdown(
        "<h2 style='font-family:Segoe UI, sans-serif; font-size:26px; font-weight:900;'>📊 Visualisation & Aperçu des données</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="tip-bubble">
            💡 <b>Conseils</b> : utilisez les filtres, survolez les graphiques pour voir les valeurs exactes, et exportez les tableaux en CSV.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================== 1) APERÇU DES DONNÉES ==================
    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
    st.markdown("### Aperçu des données")
    st.info(
        "ℹ️  **Note d’interprétation**:\n- Données et similarités**\n• Le tableau *movies* répertorie les films (identifiant, titre) et permet la recherche par titre ainsi que l’export d’un sous-ensemble.\n• La section *similarity* affiche un extrait de la matrice de similarité (scores ∈ [0,1]) limité aux N premiers films sélectionnés."
    )
    st.caption("Les variables de mesure (popularité, note, durée, genres) sont chargées à la demande depuis l’API TMDB.")

    n_preview = st.slider("Nombre de lignes à afficher (aperçu 'movies') :", 5, 50, 12, key="viz_n_preview")
    col_movies, col_sim = st.columns(2, gap="large")

    # ---------- A) APERÇU : MOVIES ----------
    with col_movies:
        st.markdown("#### 🎞️ Films (`movies`)")
        if movies is None or movies.empty:
            st.info("📦 Le tableau `movies` est vide ou indisponible.")
        else:
            q = st.text_input("🔎 Rechercher un titre :", key="viz_filter_title")
            df_show = movies if not q.strip() else movies[movies["title"].str.contains(q.strip(), case=False, na=False)]
            st.dataframe(df_show.head(n_preview), use_container_width=True)

            # Téléchargements (cap à 1000 lignes)
            csv_bytes = df_show.head(1000).to_csv(index=False).encode("utf-8-sig")
            pkl_buf = io.BytesIO()
            df_show.head(1000).to_pickle(pkl_buf)
            pkl_buf.seek(0)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇️ Télécharger (CSV)",
                    data=csv_bytes,
                    file_name="movies_preview.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_movies_csv",
                )
            with c2:
                st.download_button(
                    "⬇️ Télécharger (Pickle)",
                    data=pkl_buf,
                    file_name="movies_preview.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="dl_movies_pkl",
                )

    # ---------- B) APERÇU : SIMILARITY ----------
    with col_sim:
        st.markdown("#### 🔗 Similarités (`similarity`)")
        if similarity is None:
            st.info("📦 La matrice `similarity` est indisponible.")
        else:
            top_n = st.number_input("Taille de l’extrait (N films) :", 5, 30, 10, step=1, key="viz_sim_topn")

            if movies is None or movies.empty:
                st.info("Impossible d'annoter l’extrait sans `movies` valide.")
            else:
                idxs = list(range(min(int(top_n), len(movies))))
                titles = movies.iloc[idxs]["title"].tolist()

                @st.cache_data(show_spinner=False, ttl=900)
                def _similarity_subset(idxs_tuple):
                    if isinstance(similarity, np.ndarray):
                        return similarity[np.ix_(idxs_tuple, idxs_tuple)]
                    return [[similarity[i][j] for j in idxs_tuple] for i in idxs_tuple]

                try:
                    sub = _similarity_subset(tuple(idxs))
                    sub_df = pd.DataFrame(sub, index=titles, columns=titles)
                    st.dataframe(sub_df.head(30), use_container_width=True, height=360)
                except Exception as e:
                    st.warning(f"Affichage de l’extrait impossible : {e}")

                # Exports — uniquement l’extrait affiché
                try:
                    npy_buf = io.BytesIO()
                    np.save(npy_buf, np.array(sub, dtype="float32"))
                    npy_buf.seek(0)
                    sub_pkl = io.BytesIO()
                    pickle.dump(sub, sub_pkl, protocol=pickle.HIGHEST_PROTOCOL)
                    sub_pkl.seek(0)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            "⬇️ Similarity (NPY — extrait)",
                            data=npy_buf,
                            file_name="similarity_subset.npy",
                            mime="application/octet-stream",
                            use_container_width=True,
                            key="dl_sim_npy",
                        )
                    with c2:
                        st.download_button(
                            "⬇️ Similarity (Pickle — extrait)",
                            data=sub_pkl,
                            file_name="similarity_subset.pkl",
                            mime="application/octet-stream",
                            use_container_width=True,
                            key="dl_sim_pkl",
                        )
                except Exception:
                    pass

    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

    # ================== 2) ÉCHANTILLON ANNOTÉ (TOP-K) ==================
    st.markdown("###  Échantillon annoté — voisins les plus similaires")
    st.info(
        "ℹ️ **Note d’interprétation**:\nChoisissez **K** (nombre de voisins) et **N films** (taille de l’échantillon). Pour chaque film, on liste ses K films les plus proches. Export CSV possible."
    )

    if movies is None or movies.empty or similarity is None:
        st.info("📦 Nécessite `movies` et `similarity` valides.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            k = st.slider("K", 1, 10, 5, key="viz_topk")
        with c2:
            limit_titles = st.slider("N films", 5, 100, 20, key="viz_limit_titles")

        @st.cache_data(show_spinner=False, ttl=900)
        def build_topk_df_cached(k_val: int, limit_val: int):
            n = min(limit_val, len(movies))
            rows = []
            is_np = isinstance(similarity, np.ndarray)
            for i in range(n):
                scores = similarity[i] if is_np else similarity[i]
                order = np.argsort(-scores) if is_np else sorted(range(len(scores)), key=lambda j: -scores[j])
                order = [j for j in order if j != i][:k_val]
                base_title = movies.iloc[i]["title"]
                for rank, j in enumerate(order, start=1):
                    if j < len(movies):
                        rows.append(
                            {
                                "title": base_title,
                                "rec_rank": rank,
                                "similar_title": movies.iloc[j]["title"],
                                "score": float(scores[j]),
                            }
                        )
            return pd.DataFrame(rows, columns=["title", "rec_rank", "similar_title", "score"])

        if st.button("🧮 Générer l’échantillon", key="viz_build_topk_btn"):
            try:
                df_topk = build_topk_df_cached(int(k), int(limit_titles))
                st.dataframe(df_topk.head(50), use_container_width=True)
                topk_csv = df_topk.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ Télécharger l’échantillon (CSV)",
                    data=topk_csv,
                    file_name=f"similarity_top{k}_sample{limit_titles}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_topk_csv",
                )
            except Exception as e:
                st.error(f"Erreur génération top-K : {e}")

    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

    # ================== 3) TABLEAU DE BORD STATISTIQUES ==================
    st.markdown("###  Tableau de bord statistiques")
    st.info(
        "ℹ️ Les indicateurs résument l’échantillon (volume total, segments plus/moins populaires). La vue par catégorie agrège par Genres, Années ou Tiers de popularité."
    )

    @st.cache_data(show_spinner=False, ttl=900)
    def build_features_df_by_ids(ids_tuple):
        if "tmdb_movie_cached" not in globals():
            return pd.DataFrame()
        rows = []
        for mid in ids_tuple:
            d = tmdb_movie_cached(int(mid)) or {}
            rows.append(
                {
                    "movie_id": int(mid),
                    "title": d.get("title") or "Titre",
                    "popularity": float(d.get("popularity") or 0.0),
                    "vote": float(d.get("vote_average") or 0.0),
                    "runtime": int(d.get("runtime") or 0),
                    "year": (d.get("release_date") or "")[:4],
                    "genres": [g.get("name") for g in (d.get("genres") or [])],
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            q25 = df["popularity"].quantile(0.25)
            q75 = df["popularity"].quantile(0.75)
            df["pop_bucket"] = np.where(
                df["popularity"] <= q25,
                "Moins populaires",
                np.where(df["popularity"] >= q75, "Plus populaires", "Neutres"),
            )
        return df

    if movies is None or movies.empty:
        st.info("📦 Données `movies` indisponibles pour les indicateurs.")
        df_feat = pd.DataFrame()
    else:
        mlen = len(movies)
        upper = max(1, min(800, mlen))
        default = min(180, upper)
        ids_key = movies["movie_id"].dropna().astype(int).head(int(default)).tolist()
        df_feat = build_features_df_by_ids(tuple(ids_key))

    if df_feat.empty:
        st.info("Aucun indicateur ne peut être calculé pour l’instant.")
    else:
        try:
            q1 = df_feat["popularity"].quantile(0.25)
            q3 = df_feat["popularity"].quantile(0.75)
            st.caption(f"Échantillon pour les statistiques : {len(df_feat)} films · Q1={q1:.2f} • Q3={q3:.2f}")
        except Exception:
            pass

        total_films = int(len(df_feat))
        n_pop = int((df_feat["pop_bucket"] == "Plus populaires").sum())
        n_nonpop = int((df_feat["pop_bucket"] == "Moins populaires").sum())

        # ✅ KPI en un seul bloc — évite l’erreur removeChild
        kpi_html = f"""
<div class="kpi-strip">
  <div class="kpi-card" title="Total des films dans l’échantillon">
    <div class="kpi-v">{total_films}</div>
    <div class="kpi-l">Total films</div>
  </div>
  <div class="kpi-card" title="Nombre de films classés moins populaires (≤ Q1)">
    <div class="kpi-v">{n_nonpop}</div>
    <div class="kpi-l">Films non populaires</div>
  </div>
  <div class="kpi-card" title="Nombre de films classés plus populaires (≥ Q3)">
    <div class="kpi-v">{n_pop}</div>
    <div class="kpi-l">Films populaires</div>
  </div>
</div>
"""
        st.markdown(kpi_html, unsafe_allow_html=True)

        st.markdown("#### 🎛️ Vue par catégorie")
        cols = st.columns(2)
        with cols[0]:
            cat = st.selectbox("", ["Genres", "Années", "Popularité (tiers)"], index=0, key="viz_cat", label_visibility="collapsed")
        with cols[1]:
            metric = st.selectbox(
                "",
                ["Nombre de films", "Note moyenne", "Popularité moyenne", "Durée moyenne"],
                index=0,
                key="viz_metric",
                label_visibility="collapsed",
            )

        def _agg_df(df, cat_label, metric_label):
            if df.empty:
                return pd.DataFrame(columns=["cat", "val"])
            if cat_label == "Genres":
                rows = []
                for _, r in df.iterrows():
                    for g in (r["genres"] or []):
                        rows.append({"cat": g, "vote": r["vote"], "popularity": r["popularity"], "runtime": r["runtime"]})
                dfa = pd.DataFrame(rows)
            elif cat_label == "Années":
                dfa = df.assign(cat=df["year"].replace("", np.nan)).dropna(subset=["cat"])
            else:
                dfa = df.rename(columns={"pop_bucket": "cat"}).dropna(subset=["cat"])

            if dfa.empty:
                return pd.DataFrame(columns=["cat", "val"])

            if metric_label == "Nombre de films":
                grp = dfa.groupby("cat").size().reset_index(name="val")
            elif metric_label == "Note moyenne":
                grp = dfa.groupby("cat")["vote"].mean().reset_index(name="val")
            elif metric_label == "Popularité moyenne":
                grp = dfa.groupby("cat")["popularity"].mean().reset_index(name="val")
            else:
                grp = dfa.groupby("cat")["runtime"].mean().reset_index(name="val")
            return grp.sort_values("val", ascending=False)

        grp = _agg_df(df_feat, cat, metric)
        if grp.empty:
            st.info("Pas assez de données pour cette vue.")
        else:
            fig_bar = px.bar(grp.head(20), x="cat", y="val", color_discrete_sequence=[GOLD])
            fig_bar.update_xaxes(title="")
            fig_bar.update_yaxes(title=metric)
            st.plotly_chart(_themed(fig_bar), use_container_width=True)

            # ✅ clé stable
            show_pie = st.toggle("Afficher un diagramme en camembert", value=False, key="viz_pie_toggle")
            if show_pie:
                pie_df = grp.head(8)
                fig_pie = px.pie(pie_df, names="cat", values="val", hole=0.35, color_discrete_sequence=COLOR_SEQ)
                st.plotly_chart(_themed(fig_pie), use_container_width=True)

                tot = float(pie_df["val"].sum() or 1.0)
                top = pie_df.iloc[0]
                pct = 100.0 * float(top["val"]) / tot
                st.caption(
                    f"📊 Ce camembert montre la répartition de **{metric.lower()}** par **{cat.lower()}**. "
                    f"La plus grande part est **{top['cat']}** avec **{int(top['val'])}** ({pct:.1f} %)."
                )

    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

    # ================== 4) INSIGHTS SUPPLÉMENTAIRES ==================
    st.markdown("###  Insights supplémentaires")
    st.info(
        "ℹ️ Cette section synthétise vos interactions (« J’aime » / « Je n’aime pas »), met en évidence vos genres favoris et présente des tendances (TMDB) + une heatmap mensuelle."
    )

    def _strip_accents(s: str) -> str:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            s = ""
        try:
            s = str(s)
        except Exception:
            s = ""
        try:
            return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower().strip()
        except Exception:
            return str(s).lower().strip()

    def _ensure_genres_runtime(mid: int):
        try:
            d = tmdb_movie_cached(mid) if "tmdb_movie_cached" in globals() else {}
            d = d or {}
            gens = [g["name"] for g in d.get("genres", [])]
            runtime = int(d.get("runtime") or 0)
            if ("fetch_overview_vote_genres_fast" in globals()) and (not gens or runtime <= 0):
                ov, gtxt, data = fetch_overview_vote_genres_fast(mid)
                if not gens:
                    gens = [g.strip() for g in (gtxt or "").replace("/", ",").split(",") if g.strip()]
                if runtime <= 0:
                    runtime = int((data or {}).get("runtime") or 0)
            return gens, runtime
        except Exception:
            return [], 0

    st.session_state.setdefault("interaction_log", [])
    interactions = list(st.session_state.interaction_log)

    if not interactions:
        def _build_predefined_interactions(movies_df, n=24):
            rows = []
            if movies_df is None or movies_df.empty:
                return rows
            base = movies_df.head(n)
            now = datetime.now()
            for i, r in enumerate(base.itertuples(index=False)):
                try:
                    mid = int(getattr(r, "movie_id"))
                except Exception:
                    continue
                title = getattr(r, "title")
                ts = (now - timedelta(days=30 * (i % 6))).replace(day=1, hour=12, minute=0, second=0)
                action = "like" if (i % 4 != 3) else "dislike"
                rows.append({"mid": mid, "title": title, "action": action, "ts": ts.isoformat()})
            return rows

        interactions = _build_predefined_interactions(movies, n=24)

    if interactions:
        inter_rows = []
        for it in interactions:
            mid = int(it.get("mid") or 0)
            title = it.get("title") or "Titre"
            action = "like" if (it.get("action") or "like").lower().startswith("like") else "dislike"
            try:
                ts = datetime.fromisoformat((it.get("ts") or datetime.now().isoformat()).split(".")[0])
            except Exception:
                ts = datetime.now()
            gens, runtime = _ensure_genres_runtime(mid)
            inter_rows.append(
                {
                    "mid": mid,
                    "title": title,
                    "action": action,
                    "timestamp": ts,
                    "month": ts.strftime("%Y-%m"),
                    "genres": gens,
                    "runtime_min": int(runtime or 0),
                }
            )
        df_inter = pd.DataFrame(inter_rows)

        like_counts = {}
        for _, row in df_inter[df_inter["action"] == "like"].iterrows():
            for g in (row["genres"] or []):
                like_counts[g] = like_counts.get(g, 0) + 1
        top_genres = sorted(like_counts.items(), key=lambda x: -x[1])[:3]
        top_text = ", ".join(f"{g} ({n})" for g, n in top_genres) if top_genres else "—"

        n_likes = int((df_inter["action"] == "like").sum())
        n_dislikes = int((df_inter["action"] == "dislike").sum())

        # ✅ KPI interactions en un seul bloc
        kpi2_html = f"""
<div class="kpi-strip">
  <div class="kpi-card" title="Genres les plus choisis : {top_text}">
    <div class="kpi-v">★</div>
    <div class="kpi-l">Top genres</div>
  </div>
  <div class="kpi-card" title="Total des films aimés">
    <div class="kpi-v">{n_likes}</div>
    <div class="kpi-l">Total aimés</div>
  </div>
  <div class="kpi-card" title="Total des films non aimés">
    <div class="kpi-v">{n_dislikes}</div>
    <div class="kpi-l">Total non aimés</div>
  </div>
</div>
"""
        st.markdown(kpi2_html, unsafe_allow_html=True)

        # ---------- TENDANCES TMDB ----------
        st.subheader("🔥 Films tendances (Monde)")
        st.info(
            "ℹ️ Filtrez par **genre** et **année**. Si le flux *Popular* ne couvre pas l'année demandée, on interroge *Discover* TMDB automatiquement."
        )

        if "tmdb_popular" not in globals():
            st.info("Le flux TMDB 'popular' n’est pas disponible dans cet environnement.")
        else:
            GENRE_MAP = {
                28: "Action",
                12: "Aventure",
                16: "Animation",
                35: "Comédie",
                80: "Crime",
                99: "Documentaire",
                18: "Drame",
                10751: "Famille",
                14: "Fantastique",
                36: "Histoire",
                27: "Horreur",
                10402: "Musique",
                9648: "Mystère",
                10749: "Romance",
                878: "Science Fiction",
                10770: "Téléfilm",
                53: "Thriller",
                10752: "Guerre",
                37: "Western",
            }
            GENRE_ALIASES = {
                "horror": "Horreur",
                "horreur": "Horreur",
                "sci-fi": "Science Fiction",
                "science-fiction": "Science Fiction",
                "science fiction": "Science Fiction",
                "comedy": "Comédie",
                "drama": "Drame",
                "fantasy": "Fantastique",
                "history": "Histoire",
                "music": "Musique",
                "mystery": "Mystère",
                "tv movie": "Téléfilm",
                "war": "Guerre",
            }

            def _norm(s):
                import unicodedata
                s = (s or "").strip()
                s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
                return s.lower()

            @st.cache_data(show_spinner=False, ttl=900)
            def get_trending_pool(limit=120):
                try:
                    return tmdb_popular(limit=limit) or []
                except Exception:
                    return []

            pool = get_trending_pool(limit=120)

            # 1) Select GENRE
            genre_options = [
                "Tous",
                "Action",
                "Aventure",
                "Animation",
                "Comédie",
                "Crime",
                "Documentaire",
                "Drame",
                "Famille",
                "Fantastique",
                "Histoire",
                "Horreur",
                "Musique",
                "Mystère",
                "Romance",
                "Science Fiction",
                "Téléfilm",
                "Thriller",
                "Guerre",
                "Western",
                "Horror",
            ]
            selected_gen = st.selectbox("Genre", genre_options, index=0, key="trend_gen")

            def extract_genre_names(d, m):
                names = [g["name"] for g in (d.get("genres") or [])]
                if not names and m.get("genre_ids"):
                    for gid in m.get("genre_ids") or []:
                        if gid in GENRE_MAP:
                            names.append(GENRE_MAP[gid])
                return ", ".join(sorted(set(names)))

            base_rows = []
            for m in pool:
                mid = m.get("id")
                if not mid:
                    continue
                d = tmdb_movie_cached(mid) if "tmdb_movie_cached" in globals() else {}
                d = d or {}
                title = d.get("title") or m.get("title") or "Titre"
                pop = float(d.get("popularity") or m.get("popularity") or 0.0)
                gens = extract_genre_names(d, m)
                year = (d.get("release_date") or m.get("release_date") or "")[:4]
                base_rows.append({"title": title, "popularity": pop, "genres": gens, "year": year})

            df_all = pd.DataFrame(base_rows)

            # 2) Années DYNAMIQUES (selon le genre choisi)
            def match_genre(genres_str, sel):
                if sel in ("Tous", "", None):
                    return True
                sel_norm = _norm(GENRE_ALIASES.get(_norm(sel), sel))
                gens_norm = _norm("" if pd.isna(genres_str) else str(genres_str))
                return sel_norm in gens_norm

            if df_all.empty:
                st.info("Impossible d’afficher les tendances : résultats vides TMDB.")
            else:
                df_g = df_all[df_all["genres"].apply(lambda s: match_genre(s, selected_gen))] if selected_gen != "Tous" else df_all
                years_avail = [y for y in sorted(df_g["year"].dropna().unique().tolist(), reverse=True) if str(y).strip()]
                if not years_avail:
                    years_avail = [y for y in sorted(df_all["year"].dropna().unique().tolist(), reverse=True) if str(y).strip()]
                years_options = ["Toutes"] + years_avail
                selected_year = st.selectbox("Année (disponibles)", years_options, index=0, key="trend_year")

                # 3) Filtre principal
                df_tr = df_g.copy()
                before = len(df_tr)
                if selected_year != "Toutes":
                    df_tr = df_tr[df_tr["year"].astype(str) == str(selected_year)]
                st.caption(f"🔍 Résultats (Popular): {before} → après filtres : {len(df_tr)}")

                relaxed_msg = None

                # 4) Fallback DISCOVER si vide pour (genre + année)
                def _gid_from_name(name):
                    nm = GENRE_ALIASES.get(_norm(name), name)
                    for gid, label in GENRE_MAP.items():
                        if _norm(label) == _norm(nm):
                            return gid
                    return None

                if df_tr.empty and selected_year != "Toutes" and "tmdb_get" in globals():
                    try:
                        params = {
                            "sort_by": "popularity.desc",
                            "language": "fr-FR",
                            "page": 1,
                            "primary_release_year": int(selected_year),
                        }
                        gid = _gid_from_name(selected_gen) if selected_gen != "Tous" else None
                        if gid:
                            params["with_genres"] = gid
                        d = tmdb_get("discover/movie", params=params) or {}
                        disc_rows = []
                        for x in (d.get("results") or [])[:20]:
                            disc_rows.append(
                                {
                                    "title": x.get("title") or "Titre",
                                    "popularity": float(x.get("popularity") or 0.0),
                                    "genres": extract_genre_names({}, x),
                                    "year": (x.get("release_date") or "")[:4],
                                }
                            )
                        df_tr = pd.DataFrame(disc_rows)
                        if not df_tr.empty:
                            relaxed_msg = "Pas de résultat dans *Popular*. Affichage via **Discover** (année/genre)."
                    except Exception:
                        pass

                # 5) Dernier recours : global
                if df_tr.empty:
                    df_tr = df_all.sort_values("popularity", ascending=False).head(15)
                    relaxed_msg = (relaxed_msg or "") + " Aucun film exact. **Tendances globales**."

                if relaxed_msg:
                    st.warning(relaxed_msg.strip())

                # 6) Rendu (toujours un graphe)
                df_tr = df_tr.sort_values("popularity", ascending=False).head(15)
                fig_tr = px.bar(df_tr, x="title", y="popularity", color_discrete_sequence=[BORDEAUX], hover_data=["year", "genres"])
                fig_tr.update_xaxes(title="", tickangle=45)
                fig_tr.update_yaxes(title="Popularité (TMDB)")
                st.plotly_chart(_themed(fig_tr), use_container_width=True)
                st.dataframe(df_tr[["title", "year", "genres", "popularity"]], use_container_width=True)

                # Série par année (sur le pool filtré genre)
                try:
                    rows_year = []
                    for _, r in df_g.iterrows():
                        if r["year"] and str(r["year"]).isdigit():
                            rows_year.append({"year": int(r["year"]), "popularity": float(r["popularity"] or 0.0)})
                    if rows_year:
                        df_year = pd.DataFrame(rows_year).groupby("year")["popularity"].mean().reset_index().sort_values("year")
                        fig_year = px.line(df_year, x="year", y="popularity", markers=True, color_discrete_sequence=[GOLD])
                        fig_year.update_xaxes(title="Année (disponibles)")
                        fig_year.update_yaxes(title="Popularité moyenne (genre sélectionné)")
                        st.plotly_chart(_themed(fig_year), use_container_width=True)
                except Exception:
                    pass

        # ---------- HEATMAP ----------
        st.subheader("⌛ Heatmap — Films les plus regardés (par mois)")
        st.info("ℹ️ On regroupe vos **likes** par *mois × titre* et on affiche une **heatmap** ou des **barres**.")
        df_like = df_inter[df_inter["action"] == "like"].copy()
        if df_like.empty:
            st.caption("Pas assez de likes pour construire la heatmap.")
        else:
            counts = df_like.groupby(["month", "title"]).size().reset_index(name="n")
            top_titles = counts.groupby("title")["n"].sum().sort_values(ascending=False).head(12).index.tolist()
            counts = counts[counts["title"].isin(top_titles)].copy()
            try:
                counts["ym"] = pd.to_datetime(counts["month"] + "-01")
            except Exception:
                counts["ym"] = pd.to_datetime(counts["month"], errors="coerce")
            counts = counts.dropna(subset=["ym"]).assign(month_label=lambda d: d["ym"].dt.strftime("%Y-%m"))
            last_12 = sorted(counts["month_label"].unique())[-12:]
            counts = counts[counts["month_label"].isin(last_12)]
            if counts.empty:
                st.caption("Aucune donnée suffisante sur les 12 derniers mois.")
            else:
                pivot = counts.pivot_table(index="month_label", columns="title", values="n", aggfunc="sum").fillna(0)
                pivot = pivot.sort_index()

                as_bars = st.toggle("Afficher en barres (plus lisible)", value=False, key="hm_as_bars")
                if as_bars:
                    long_df = counts[counts["title"].isin(top_titles)].copy()
                    fig_barhm = px.bar(long_df, x="month_label", y="n", color="title", barmode="group", color_discrete_sequence=COLOR_SEQ)
                    fig_barhm.update_xaxes(title="Mois")
                    fig_barhm.update_yaxes(title="Likes")
                    st.plotly_chart(_themed(fig_barhm), use_container_width=True)
                else:
                    z = pivot.values
                    fig_hm2 = go.Figure(
                        data=go.Heatmap(
                            z=z,
                            x=pivot.columns.tolist(),
                            y=pivot.index.tolist(),
                            colorscale=HEATMAP_SCALE,
                            colorbar=dict(
                                title=dict(text="Likes", font=dict(color=BLANC)),
                                tickcolor=BLANC,
                                tickfont=dict(color=BLANC),
                            ),
                            xgap=3,
                            ygap=3,
                            showscale=True,
                        )
                    )
                    fig_hm2.update_xaxes(title="Titre", showgrid=False)
                    fig_hm2.update_yaxes(title="Mois", showgrid=False)
                    st.plotly_chart(_themed(fig_hm2), use_container_width=True)



#----- PARTIE 3-------
# -------- RECOMMANDATION --------
elif page == "🎬 Recommandation":
    import re, pickle, numpy as np, pandas as pd, urllib.parse as _url

    st.markdown(
        "<h2 style='font-family:Segoe UI, sans-serif; font-size:30px; font-weight:900;'>🎬 Système de recommandation de films</h2>",
        unsafe_allow_html=True
    )

    # ---- Styles UI (images plus nettes + cartes) ----
    st.markdown("""
    <style>
      .card { border:1px solid rgba(255,215,0,0.25); border-radius:16px;
        padding:12px; background:#1a1a1a; margin-bottom:12px; }
      .title-big{font-family:Segoe UI, sans-serif; font-size:22px; font-weight:900; margin:4px 0 8px 0;}
      .badge-genre{display:inline-block; padding:2px 10px; border-radius:999px;
        background:rgba(255,215,0,0.12); color:#FFD700; font-size:13px; margin:0 6px 8px 0;}
      .meta-line{font-size:15px; color:rgba(255,255,255,0.95); margin:6px 0; font-weight:700;}
      .note-xxl{ display:inline-block; padding:8px 14px; border:2px solid rgba(255,215,0,0.6);
        border-radius:14px; color:#FFD700; background:rgba(255,215,0,0.08);
        font-weight:900; letter-spacing:.5px; font-size:28px; line-height:1; margin:6px 0 10px 0; }
      .poster img{ image-rendering:auto; border-radius:12px; }
      .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap:12px; }
      .grid .it { background:#121212; border:1px solid rgba(230,196,110,0.22); border-radius:14px; padding:8px; }
      .grid .cap { font-weight:800; margin-top:6px; font-size:13.5px; }
    </style>
    """, unsafe_allow_html=True)

    # ---- Utilitaires ----
    def _as_int(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    def minutes_to_hhmm(m):
        try:
            m = int(m or 0)
        except Exception:
            m = 0
        if m <= 0: return "Durée indisponible"
        h, mn = divmod(m, 60)
        return f"{h} h {mn:02d}"

    def build_watch_link_fr(title: str, year: str | int | None = None) -> str:
        base_query = f"{title} {year}" if year else title
        q_yt = f"{base_query} vf français film complet"
        return f"https://www.youtube.com/results?search_query={_url.quote_plus(q_yt)}&hl=fr"

    def _safe_fetch_poster(mid, size="w500"):
        try:
            if 'fetch_poster' in globals():
                return fetch_poster(mid, size=size)
            if 'fetch_poster_fast' in globals():
                return fetch_poster_fast(mid, size=size)
        except Exception:
            pass
        return None

    # ---- Chargement modèles/données (content + collaboratif) ----
    @st.cache_resource
    def _load_models_all():
        # Content
        try:
            movies = pickle.load(open("movie_list.pkl", "rb"))
            similarity = pickle.load(open("similarity.pkl", "rb"))
        except Exception:
            movies, similarity = None, None
        # SVD
        try:
            svd_model = pickle.load(open("svd_model.pkl", "rb"))
            svd_items = pickle.load(open("svd_items.pkl", "rb"))
        except Exception:
            svd_model, svd_items = None, []
        # ALS
        try:
            als_item_f = np.load("als_item_factors.npy")
            als_user_f = np.load("als_user_factors.npy")
            als_items_map = pickle.load(open("als_items.pkl", "rb"))  # idx_item -> movie_id (TMDB)
            # normaliser les clés/valeurs en int
            als_items_map = {
                _as_int(k): _as_int(v)
                for k, v in (als_items_map or {}).items()
                if _as_int(k) is not None and _as_int(v) is not None
            }
        except Exception:
            als_item_f, als_user_f, als_items_map = None, None, {}
        return movies, similarity, svd_model, svd_items, als_item_f, als_user_f, als_items_map

    movies, similarity, svd_model, svd_items, als_item_f, als_user_f, als_items_map = _load_models_all()

    # ---- Validations robustes ----
    def _valid_content():
        return (movies is not None) and (similarity is not None) and (not getattr(movies, "empty", True)) and (len(similarity) == len(movies))

    def _valid_als():
        if (als_item_f is None) or (als_user_f is None) or (not als_items_map):
            return False
        # vecteur utilisateur cohérent
        if als_user_f.ndim == 1:
            u = als_user_f
        else:
            u = als_user_f[0]
        return (als_item_f.ndim == 2) and (u.shape[0] == als_item_f.shape[1])

    # ---- LUTs id→titre + validité ----
    @st.cache_resource
    def _build_id_luts(movies_df):
        if movies_df is None or getattr(movies_df, "empty", True):
            return {}, set()
        m = movies_df.copy()
        ids = pd.to_numeric(m["movie_id"], errors="coerce").fillna(-1).astype(int)
        titles = m["title"].astype(str)
        id2title = dict(zip(ids.tolist(), titles.tolist()))
        return id2title, set(id2title.keys())

    ID2TITLE, VALID_IDS = _build_id_luts(movies)

    def resolve_title(mid: int) -> str:
        if mid in ID2TITLE:
            return ID2TITLE[mid]
        try:
            d = tmdb_movie_cached(mid) if 'tmdb_movie_cached' in globals() else {}
            t = (d or {}).get("title")
            if t: return str(t)
        except Exception:
            pass
        return f"Film #{mid}"

    def resolve_details(mid: int) -> dict:
        base = {"title": resolve_title(mid)}
        details = {}
        if 'tmdb_movie_cached' in globals():
            try: details = tmdb_movie_cached(mid) or {}
            except Exception: details = {}
        if (not details) and ('tmdb_get' in globals()):
            try: details = tmdb_get(f"movie/{mid}", params={"append_to_response": "release_dates"}) or {}
            except Exception: details = {}

        title = details.get("title") or base.get("title")
        overview = details.get("overview") or ""
        vote = float(details.get("vote_average") or 0.0)
        runtime_min = int(details.get("runtime") or 0)
        release_date = details.get("release_date") or ""
        year = release_date.split("-")[0] if release_date else ""
        countries = ", ".join([c.get("name", "") for c in details.get("production_countries", [])]) or "Pays indisponible"

        genres_text = ""
        try:
            if details.get("genres"):
                genres_text = ", ".join(sorted({g.get("name", "") for g in details["genres"] if g.get("name")}))
            elif details.get("genre_ids"):
                genres_text = ", ".join(str(g) for g in details["genre_ids"])
        except Exception:
            pass

        return {
            "title": title, "overview": overview, "vote": vote,
            "genres_text": genres_text or "Genres indisponibles",
            "runtime_min": runtime_min, "year": year, "countries": countries,
            "raw": details
        }

    # ---- Recherche index par titre (tolérante) ----
    def _find_title_index(mdf: pd.DataFrame, title: str):
        if mdf is None or getattr(mdf, "empty", True) or not title:
            return None
        # 1) exact
        m = mdf.index[mdf["title"] == title]
        if len(m) > 0: return int(m[0])
        # 2) caseless
        tcf = str(title).casefold().strip()
        m2 = mdf.index[mdf["title"].astype(str).str.casefold().str.strip() == tcf]
        if len(m2) > 0: return int(m2[0])
        # 3) contient (début/fin)
        esc = re.escape(title.strip())
        m3 = mdf.index[mdf["title"].astype(str).str.contains(esc, case=False, regex=True)]
        if len(m3) > 0: return int(m3[0])
        # 4) fallback : premier film
        try:
            return int(mdf.index[0])
        except Exception:
            return None

    # ---- Content-based (cosine) avec fallback si vide ----
    @st.cache_data(show_spinner=False, ttl=900)
    def _content_ids_by_title_cached(title: str, k: int, same_genre_only: bool = True):
        if not _valid_content():
            return []
        idx = _find_title_index(movies, title)
        if idx is None:
            return []
        scores = list(enumerate(similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx]

        mids_ranked = []
        base_genres = set()
        try:
            base_genres = set(str(movies.iloc[idx].get("genres","")).lower().replace("/", ",").split(","))
            base_genres = {g.strip() for g in base_genres if g.strip()}
        except Exception:
            pass

        for i, _ in scores:
            mid = _as_int(movies.iloc[i].get("movie_id"))
            if mid is None: 
                continue
            if same_genre_only and base_genres:
                try:
                    g = set(str(movies.iloc[i].get("genres","")).lower().replace("/", ",").split(","))
                    g = {x.strip() for x in g if x.strip()}
                    if not (g & base_genres):
                        continue
                except Exception:
                    pass
            mids_ranked.append(mid)

        mids_ranked = [m for m in dict.fromkeys(mids_ranked) if m in VALID_IDS]
        if not mids_ranked and same_genre_only:
            # >>> Fallback : sans filtre de genre pour éviter le "aucun film"
            return _content_ids_by_title_cached(title, k=k, same_genre_only=False)
        return mids_ranked[:k]

    # ---- Collaboratif SVD ----
    def recommend_svd_for_user(k: int = 6) -> list[int]:
        if svd_model is None or not svd_items:
            return []
        uid = "local_user"
        scored = []
        for mid in svd_items:
            mi = _as_int(mid)
            if mi is None: 
                continue
            try:
                est = float(svd_model.predict(uid, mi).est)
                scored.append((mi, est))
            except Exception:
                continue
        scored.sort(key=lambda x: x[1], reverse=True)
        mids = [m for m, _ in scored]
        mids = [m for m in dict.fromkeys(mids) if m in VALID_IDS]
        return mids[:k]

    # ---- Collaboratif ALS (avec validations + fallback) ----
    def recommend_als_for_seed(seed_mid: int | None, k: int = 6) -> list[int]:
        if not _valid_als():
            return []
        # vecteur utilisateur
        uvec = als_user_f if als_user_f.ndim == 1 else als_user_f[0]
        # scores items
        try:
            scores = als_item_f @ uvec  # (n_items,)
        except Exception:
            return []
        order = np.argsort(-scores)

        mids = []
        seen_mids = set()
        for idx_item in order:
            idx_item = int(idx_item)
            mid = als_items_map.get(idx_item)
            if mid is None:
                continue
            if mid in seen_mids:
                continue
            if (seed_mid is not None) and (mid == seed_mid):
                continue
            if (VALID_IDS and mid not in VALID_IDS):
                continue
            seen_mids.add(mid)
            mids.append(mid)
            if len(mids) >= k:
                break

        # Fallback si vide/insuffisant : on complète via contenu autour du seed
        need = max(0, k - len(mids))
        if need > 0:
            ref_title = resolve_title(seed_mid) if seed_mid else None
            if ref_title:
                cb = _content_ids_by_title_cached(ref_title, k=k*2, same_genre_only=True)
                for m in cb:
                    if m not in mids and ((not seed_mid) or m != seed_mid):
                        mids.append(m)
                        if len(mids) >= k:
                            break
        return mids[:k]

    # ---- Session: likes partagés avec "Recommandation avancée" ----
    if "liked_movies" not in st.session_state:
        st.session_state["liked_movies"] = []
    liked_mids: list[int] = [ _as_int(x) for x in st.session_state["liked_movies"] if _as_int(x) is not None ]
    liked_mids = [m for m in dict.fromkeys(liked_mids) if (m is not None and (not VALID_IDS or m in VALID_IDS))]

    # ===================== UI PRINCIPALE =====================
    if not _valid_content():
        st.info("📦 Données indisponibles ou non alignées. Assure-toi que 'movie_list.pkl' et 'similarity.pkl' correspondent au même corpus.")
        st.stop()

    # Sélecteur film + mode (sans case “même genre”)
    csel, cmode = st.columns([3, 2])
    with csel:
        selected_movie = st.selectbox("🎯 Choisissez un film :", movies["title"].values, key="reco_movie_sel")
    with cmode:
        options = ["Content-based (cosine)"]
        if (svd_model is not None) and svd_items:
            options += ["Collaboratif (SVD)"]
        if _valid_als():
            options += ["Collaboratif (ALS)"]
        mode = st.selectbox("Mode de recommandation", options, index=0, key="reco_mode_sel")

    # Afficher vos likes (info, pas de boutons ici)
    if liked_mids:
        st.markdown("### 🧡 Vos likes (synchro depuis *Recommandation avancée*)")
        st.markdown('<div class="grid">', unsafe_allow_html=True)
        for mid in liked_mids[:12]:
            poster = _safe_fetch_poster(mid, size="w500")
            st.markdown('<div class="it">', unsafe_allow_html=True)
            if poster:
                st.image(poster, width=180, use_container_width=False)
            st.markdown(f'<div class="cap">{resolve_title(mid)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Bouton principal
    if st.button("🔁 Recommander", key="btn_reco_run"):
        # index + mid sélectionné (robuste)
        idx_sel = _find_title_index(movies, selected_movie)
        selected_mid = None
        try:
            if idx_sel is not None:
                selected_mid = _as_int(movies.iloc[idx_sel]["movie_id"])
        except Exception:
            selected_mid = None

        # Recos principales (6) avec fallback durs
        mids_main = []
        if mode == "Content-based (cosine)":
            mids_main = _content_ids_by_title_cached(selected_movie, k=6, same_genre_only=True)
            if not mids_main:
                mids_main = _content_ids_by_title_cached(selected_movie, k=6, same_genre_only=False)
        elif mode == "Collaboratif (SVD)":
            mids_main = recommend_svd_for_user(k=6)
            if not mids_main:
                # fallback sur content
                mids_main = _content_ids_by_title_cached(selected_movie, k=6, same_genre_only=False)
        else:  # ALS
            seed = (liked_mids[-1] if liked_mids else selected_mid)
            mids_main = recommend_als_for_seed(seed, k=6)
            if not mids_main:
                # fallback → SVD → content
                mids_main = recommend_svd_for_user(k=6) or _content_ids_by_title_cached(resolve_title(seed) if seed else selected_movie, k=6, same_genre_only=False)

        # Nettoyage final
        mids_main = [ _as_int(m) for m in (mids_main or []) if _as_int(m) is not None ]
        if selected_mid is not None:
            mids_main = [m for m in mids_main if m != selected_mid]
        if VALID_IDS:
            mids_main = [m for m in mids_main if m in VALID_IDS]
        # uniques et tranche
        mids_main = list(dict.fromkeys(mids_main))[:6]

        if not mids_main:
            st.warning("Aucune recommandation exploitable avec les modèles actuels.")
        else:
            for mid in mids_main:
                meta = resolve_details(mid)
                title_txt   = meta["title"]
                overview    = meta["overview"]
                vote        = meta["vote"]
                genres_text = meta["genres_text"]
                runtime_min = meta["runtime_min"]
                runtime_hr  = minutes_to_hhmm(runtime_min)
                year        = meta["year"]
                countries   = meta["countries"]

                # Certification d'âge (si dispo)
                age_cert = "Accès tout public"
                try:
                    rels = (meta["raw"] or {}).get("release_dates", {}).get("results", [])
                    def pick_cert(cc):
                        for r in rels:
                            if r.get("iso_3166_1") == cc:
                                for c in r.get("release_dates", []):
                                    cert = (c.get("certification") or "").strip()
                                    if cert: return cert
                        return None
                    age_cert = pick_cert("FR") or pick_cert("US") or "Accès tout public"
                except Exception:
                    pass

                poster = _safe_fetch_poster(mid, size="w500")
                watch_url = build_watch_link_fr(title_txt, year)

                c1, c2 = st.columns([1.05, 2])
                with c1:
                    if poster:
                        st.image(poster, caption=f"**{title_txt}**", width=220, use_container_width=False)
                    else:
                        st.write(f"**{title_txt}**")
                with c2:
                    st.markdown(f"<div class='title-big'>{title_txt}</div>", unsafe_allow_html=True)
                    st.markdown(f"<span class='badge-genre'>{genres_text}</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='note-xxl'><span style='font-size:14px;font-weight:800;opacity:.9;margin-right:8px;'>NOTE</span> {vote:.1f}</div>", unsafe_allow_html=True)
                    meta_bits = []
                    if runtime_min:  meta_bits.append(f"🕒 <b>{runtime_hr}</b>")
                    if year:         meta_bits.append(f"📅 <b>{year}</b>")
                    if countries:    meta_bits.append(f"🌍 <b>{countries}</b>")
                    if meta_bits:
                        st.markdown(f"<div class='meta-line'>{' • '.join(meta_bits)}</div>", unsafe_allow_html=True)
                    st.markdown("**Synopsis**")
                    st.write(overview or "Synopsis indisponible.")
                    st.markdown(f"[▶️ Voir le film / Plus d'infos]({watch_url})")

        # ====== BLOCS SUPPLÉMENTAIRES PILOTÉS PAR LES LIKES (sans boutons) ======
        if liked_mids:
            st.markdown("---")
            # 1) Films similaires (à partir du dernier like)
            seed_like = liked_mids[-1]
            seed_title = resolve_title(seed_like)
            if seed_title:
                st.markdown(f"### 🎞️ Films similaires à **{seed_title}**")
                sim_ids = _content_ids_by_title_cached(seed_title, k=8, same_genre_only=True) or \
                          _content_ids_by_title_cached(seed_title, k=8, same_genre_only=False)
                if sim_ids:
                    st.markdown('<div class="grid">', unsafe_allow_html=True)
                    for mid in sim_ids:
                        poster = _safe_fetch_poster(mid, size="w500")
                        st.markdown('<div class="it">', unsafe_allow_html=True)
                        if poster:
                            st.image(poster, width=180, use_container_width=False)
                        st.markdown(f'<div class="cap">{resolve_title(mid)}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Pas assez de similarités pour ce titre.")

            # 2) Catégorie de films pour vous (genres agrégés des likes)
            try:
                like_genres = set()
                for lm in liked_mids:
                    row = movies.loc[movies["movie_id"] == lm]
                    if not row.empty:
                        gtxt = str(row.iloc[0].get("genres","")).lower().replace("/", ",")
                        like_genres |= {p.strip() for p in gtxt.split(",") if p.strip()}
                if like_genres:
                    st.markdown("### 📂 Catégorie de films pour vous")
                    mask = movies["genres"].astype(str).str.lower().apply(
                        lambda g: any(gen in g for gen in like_genres)
                    )
                    subset = movies.loc[mask].copy()
                    subset = subset[~subset["movie_id"].astype(int).isin(liked_mids)]
                    subset = subset.sample(min(12, len(subset))) if len(subset) else subset

                    if not subset.empty:
                        st.markdown('<div class="grid">', unsafe_allow_html=True)
                        for _, row in subset.iterrows():
                            mid = _as_int(row["movie_id"])
                            if mid is None: continue
                            poster = _safe_fetch_poster(mid, size="w500")
                            st.markdown('<div class="it">', unsafe_allow_html=True)
                            if poster:
                                st.image(poster, width=180, use_container_width=False)
                            st.markdown(f'<div class="cap">{resolve_title(mid)}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("Pas de titres trouvés sur vos catégories préférées.")
            except Exception:
                pass

    st.stop()



#----------PARTIE 4------
# -------- PARTIE ML --------
elif page == "📈 Prédiction Machine Learning":
    st.markdown("## 📈 Analyse intelligente des films : Explorez, notez et comparez")
    st.caption("Découvrez les tendances cachées du cinéma : recherchez un film, voyez la note, la popularité et la durée — puis explorez les plus populaires comme les perles discrètes.")

    import html
    import pandas as pd  # nécessaire plus bas

    # ============== CSS (badge étoiles bas-gauche, meta sous le titre, hover, tailles uniformes) ==============
    st.markdown("""
    <style>
      :root{ --ml-poster-w:320px; } /* ajuste 300–340 selon goût */
      .genre-badge-circle{ display:none !important; } /* tue toute bulle dorée résiduelle */

      .ml-card{ width:var(--ml-poster-w); margin:12px auto 28px; text-align:center; }
      .ml-title{
        margin:12px 0 4px 0; font-weight:800; line-height:1.15;
        display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
      }
      .ml-meta{
        display:inline-flex; gap:10px; align-items:center; justify-content:center;
        font-weight:800; font-size:13px; color:#fff; opacity:.95;
        padding:6px 10px; border-radius:10px; background:rgba(0,0,0,.28);
        border:1px solid rgba(255,215,0,.25);
      }
      .ml-genre-chip{
        display:inline-block; margin:8px 0 4px 0; padding:3px 10px; border-radius:999px;
        background:rgba(255,215,0,0.12); color:#FFD700; font-size:12px;
        border:1px solid rgba(255,215,0,0.35);
        max-width:var(--ml-poster-w); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
      }

      /* 🔥 IMPORTANT : pas de coupe du synopsis (on montre un bref résumé calculé) */
      .ml-desc{
        color:rgba(255,255,255,0.92); font-size:13.5px; margin-top:6px;
        display:block; white-space:normal; overflow:visible;
      }

      .ml-poster{
        position:relative; width:var(--ml-poster-w); aspect-ratio:2/3;
        border-radius:14px; overflow:hidden; margin:0 auto; background:#000;
        border:1px solid rgba(255,215,0,0.25); box-shadow:0 6px 16px rgba(0,0,0,0.25);
      }
      .ml-poster img{
        width:100%; height:100%; object-fit:cover; display:block;
        transform-origin:center center; transition:transform .18s ease;
      }
      .ml-poster:hover img{ transform:scale(1.05); } /* hover doux */

      /* ⭐ badge en bas-gauche (comme ta capture) */
      .star-badge{
        position:absolute; left:10px; bottom:12px; z-index:2;
        font-size:14px; color:#FFD700; font-weight:900;
        background:rgba(0,0,0,0.55); border:1px solid rgba(255,215,0,0.5);
        border-radius:12px; padding:3px 8px; text-shadow:0 1px 2px rgba(0,0,0,.55);
      }
    </style>
    """, unsafe_allow_html=True)

    # ----------------- Utils -----------------
    def stars_from_vote(v):
        try: v = float(v)
        except Exception: v = 0.0
        filled = int(round(max(0.0, min(10.0, v)) / 2.0))
        return "★" * filled + "☆" * (5 - filled)

    def _synopsis_brief_or_full(txt, short_threshold=320, brief_len=240):
        """
        - Si le synopsis est court (<= short_threshold), on l'affiche EN ENTIER.
        - S'il est long, on renvoie un RÉSUMÉ bref (brief_len) en coupant proprement sur la ponctuation.
        """
        if not txt:
            return "Synopsis indisponible."
        t = " ".join(str(txt).split())
        if len(t) <= short_threshold:
            return t
        cut = max(
            t.rfind(". ", 0, brief_len),
            t.rfind("! ", 0, brief_len),
            t.rfind("? ", 0, brief_len),
            t.rfind(", ", 0, brief_len),
        )
        if cut == -1:
            cut = brief_len
        return t[:cut].rstrip(" ,;:")

    def minutes_to_hhmm(m):
        try: m = int(m or 0)
        except Exception: m = 0
        if m <= 0: return "Durée indisponible"
        h = m // 60; mn = m % 60
        return f"{h} h {mn:02d}"

    def format_popularity(x):
        try: x = float(x or 0)
        except Exception: x = 0.0
        if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
        if x >= 1_000:     return f"{x/1_000:.1f}K"
        return f"{int(round(x))}"

    # ⚙️ cache long pour l’âge (appel coûteux)
    @st.cache_data(show_spinner=False, ttl=86400)
    def get_age_certification(mid):
        try:
            details = tmdb_get(f"movie/{mid}", params={"append_to_response":"release_dates"}) or {}
            rels = details.get("release_dates", {}).get("results", [])
            def pick(cc):
                for r in rels:
                    if r.get("iso_3166_1") == cc:
                        for c in r.get("release_dates", []):
                            cert = (c.get("certification") or "").strip()
                            if cert: return cert
                return None
            return pick("FR") or pick("US") or "Tous publics"
        except Exception:
            return "Tous publics"

    # ============== Recherche ==============
    with st.expander("🔎 Recherchez un film et découvrez ce qu'en pense le public.", expanded=True):
        q = st.text_input("Quel film souhaitez-vous explorer ?", key="rating_search_title")
        if st.button("🔍 Lancer la recherche", key="btn_rating_search"):
            if not q.strip():
                st.warning("Veuillez saisir un titre de film.")
            else:
                res = tmdb_search_title(q.strip())
                if not res:
                    st.info("Aucun film correspondant trouvé.")
                else:
                    cols = st.columns(3)
                    for i, m in enumerate(res[:6]):
                        info = tmdb_details_full(m["id"])
                        poster = fetch_poster_fast(m['id'], size="w342")   # plus léger que w500
                        stars  = stars_from_vote(info["vote"])
                        run_hh = minutes_to_hhmm(info.get("runtime"))
                        age    = get_age_certification(m["id"])
                        pop    = format_popularity(info.get("popularity"))

                        with cols[i % 3]:
                            st.markdown("<div class='ml-card'>", unsafe_allow_html=True)
                            st.markdown(
                                f"""
                                <div class="ml-poster">
                                  <img src="{poster}" alt="{html.escape(info['title'])}" loading="lazy">
                                  <div class="star-badge">{stars}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.markdown(f"<div class='ml-title'>{html.escape(info['title'])}</div>", unsafe_allow_html=True)
                            st.markdown(
                                f"<div class='ml-meta'>⭐ {info['vote']:.1f} • 🔥 {pop} • 🕒 {run_hh} • 👤 {age}</div>",
                                unsafe_allow_html=True
                            )
                            st.markdown(f"<div class='ml-genre-chip'>{html.escape(info['genres'])}</div>", unsafe_allow_html=True)
                            # ⬇️ synopsis complet si court, sinon bref résumé propre
                            st.markdown(f"<div class='ml-desc'><em>{html.escape(_synopsis_brief_or_full(info['overview']))}</em></div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

    # ============== Les films qui font vibrer (optimisé & mis en cache) ==============
    import html
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # --- Helpers de cache légers (si non définis déjà) ---
    @st.cache_data(show_spinner=False, ttl=900)
    def _build_popularity_df_cached(movies_df, nmax=800):
        """Extrait (title, movie_id, popularity) à partir de tmdb_movie_cached pour un sous-ensemble."""
        rows = []
        if movies_df is None or movies_df.empty:
            return pd.DataFrame(columns=["title", "movie_id", "popularity"])
        mview = movies_df.head(min(int(nmax), len(movies_df)))
        for _, r in mview.iterrows():
            try:
                mid = int(r.get("movie_id"))
            except Exception:
                continue
            d = tmdb_movie_cached(mid) or {}
            pop = d.get("popularity", 0) or 0
            title = r.get("title", "Titre inconnu")
            rows.append((title, mid, pop))
        df = pd.DataFrame(rows, columns=["title", "movie_id", "popularity"]).drop_duplicates(subset=["movie_id"])
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
        return df

    @st.cache_data(show_spinner=False, ttl=900)
    def _pop_quantiles(df_pop):
        if df_pop.empty:
            return 0.0, 0.0, 0.0
        return (
            float(df_pop["popularity"].quantile(0.33)),
            float(df_pop["popularity"].quantile(0.66)),
            float(df_pop["popularity"].median()),
        )

    @st.cache_data(show_spinner=False, ttl=86400)
    def _age_cert(mid: int) -> str:
        try:
            return get_age_certification(mid)
        except Exception:
            return "Tous publics"

    @st.cache_data(show_spinner=False, ttl=3600)
    def _poster_url(mid: int, size: str = "w185") -> str:
        try:
            return fetch_poster_fast(mid, size=size)
        except Exception:
            return ""

    @st.cache_data(show_spinner=False, ttl=900)
    def _details_fast(mid: int) -> dict:
        try:
            ov, genres_text, data = fetch_overview_vote_genres_fast(mid)
            d = tmdb_movie_cached(mid) or {}
            return {
                "overview": ov,
                "genres_text": genres_text,
                "vote": float(data.get("vote_average") or d.get("vote_average") or 0.0),
                "runtime": int(data.get("runtime") or d.get("runtime") or 0),
                "popularity": float(d.get("popularity") or 0.0),
            }
        except Exception:
            return {"overview": "", "genres_text": "", "vote": 0.0, "runtime": 0, "popularity": 0.0}

    # --- Hydratation par lot (parallélisée) ---
    @st.cache_data(show_spinner=False, ttl=900)
    def _hydrate_batch(mids: tuple, poster_size: str = "w185") -> dict:
        results = {}
        mids_unique = list(dict.fromkeys(int(m) for m in mids if m))[:60]

        def _one(mid: int):
            p = _poster_url(mid, poster_size)
            d = _details_fast(mid)
            a = _age_cert(mid)
            return mid, {"poster": p, "details": d, "age": a}

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(_one, mid) for mid in mids_unique]
            for fut in as_completed(futures):
                try:
                    mid, payload = fut.result()
                    results[mid] = payload
                except Exception:
                    pass
        return results

    # --- UI principale ---
    if movies is None or movies.empty:
        st.info("Données de films indisponibles.")
    else:
        st.markdown("### 🔥 Les films qui font vibrer (ou pas)")
        st.caption("Des blockbusters incontournables aux trésors méconnus, sélectionnez un angle et laissez-vous surprendre.")

        classement = st.selectbox(
            "Choisissez une catégorie de popularité :",
            ["Top des films plus populaires", "Top des films neutres", "Top des films moins populaires"]
        )
        limit = st.slider("Nombre de films à afficher :", 3, 30, 12)

        df_pop = _build_popularity_df_cached(movies)
        if df_pop.empty:
            st.info("Aucune donnée de popularité utilisable.")
            st.stop()

        p33, p66, med = _pop_quantiles(df_pop)

        if classement == "Top des films plus populaires":
            subset = df_pop.sort_values("popularity", ascending=False).head(limit)
        elif classement == "Top des films neutres":
            midband = df_pop[(df_pop["popularity"] >= p33) & (df_pop["popularity"] <= p66)].copy()
            if midband.empty:
                tmp = df_pop.copy(); tmp["dist"] = (tmp["popularity"] - med).abs()
                subset = tmp.sort_values("dist").head(limit)
            else:
                midband["dist"] = (midband["popularity"] - med).abs()
                subset = midband.sort_values("dist").head(limit)
        else:
            lowband = df_pop[df_pop["popularity"] <= p33].copy()
            subset = lowband.sort_values("popularity").head(limit)

        mids = tuple(int(x) for x in subset["movie_id"].tolist())
        batch = _hydrate_batch(mids, poster_size="w185")

        ncols = min(4, max(1, limit))
        cols = st.columns(ncols)

        for i, row in enumerate(subset.reset_index(drop=True).itertuples(index=False)):
            mid = int(row.movie_id)
            payload = batch.get(mid, {})
            poster = payload.get("poster") or ""
            det = payload.get("details") or {}
            age = payload.get("age") or "Tous publics"

            vote = float(det.get("vote") or 0.0)
            stars = "★" * int(round(max(0.0, min(10.0, vote)) / 2.0)) + "☆" * (5 - int(round(max(0.0, min(10.0, vote)) / 2.0)))
            runtime = det.get("runtime") or 0
            run_hh = (lambda m: "Durée indisponible" if not int(m or 0) else f"{int(m)//60} h {int(m)%60:02d}")(runtime)
            pop = (lambda x: f"{x/1_000_000:.1f}M" if x >= 1_000_000 else (f"{x/1_000:.1f}K" if x >= 1_000 else f"{int(round(x))}"))(float(row.popularity or 0))
            genres_text = det.get("genres_text") or "Genres indisponibles"
            overview = det.get("overview") or "Description indisponible."
            short_desc = _synopsis_brief_or_full(overview)  # ⬅️ même logique que PARTIE 5

            with cols[i % ncols]:
                st.markdown("<div class='ml-card'>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="ml-poster">
                      <img src="{poster}" alt="{html.escape(row.title)}" loading="lazy">
                      <div class="star-badge">{stars}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(f"<div class='ml-title'>{html.escape(row.title)}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='ml-meta'>⭐ {vote:.1f} • 🔥 {pop} • 🕒 {run_hh} • 👤 {age}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"<div class='ml-genre-chip'>{html.escape(genres_text)}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='ml-desc'><em>{html.escape(short_desc)}</em></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)



#---------PARTIE 5--------
# -------- PARTIE RECOMMANDTAION AVANCEE --------
elif page == "🧠 Recommandation avancée":
    import html, os, json, time, numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as _norm

    # ============================
    #       ENTÊTE & STYLES
    # ============================
    st.markdown("## 🧠 Recommandation avancée — Système de recommandation intelligente")
    st.caption("Affinez vos goûts, likez/masquez, et découvrez ce qui cartonne par continent ")

    # Afficher les tendances même sans dataset local
    HAS_MOVIES = not (movies is None or getattr(movies, "empty", True))
    if not HAS_MOVIES:
        st.info("📦 Données locales indisponibles pour les SUGGESTIONS. Les TENDANCES restent affichées ci-dessous.")

    st.markdown("""
    <style>
      :root{ --adv-w:240px; }
      .genre-badge-circle{ display:none !important; }
      .adv-card{ width:var(--adv-w); margin:14px auto 26px; text-align:left; display:flex; flex-direction:column; }
      .adv-card, .adv-card * { font-family:"Times New Roman", Times, serif !important; }
      .section-lead{ margin:4px 0 10px 0; opacity:.9; }
      .flash-pill{
        display:inline-flex; align-items:center; gap:10px;
        padding:8px 12px; border-radius:999px; margin:6px 0 12px 0;
        border:1px solid rgba(34,197,94,.4); background:rgba(34,197,94,.12);
        color:#34d399; font-weight:900;
      }

      /* Bulle dorée (raison) */
      .why-pill{
        display:inline-flex; align-items:center; gap:8px; flex-wrap:wrap;
        background:rgba(255,215,0,0.10); color:#FFD700; font-weight:900;
        border:1px solid rgba(255,215,0,0.45); border-radius:999px;
        padding:6px 12px; font-size:14px; margin:8px 0 12px 0;
      }
      .why-pill .chip{
        display:inline-block; padding:2px 10px; border-radius:999px;
        background:rgba(255,215,0,0.12); color:#FFD700;
        border:1px solid rgba(255,215,0,0.35); font-weight:800; font-size:13px;
      }

      /* Poster + HOVER */
      .adv-poster{
        position:relative; width:var(--adv-w); aspect-ratio:2/3;
        border-radius:14px; overflow:hidden; margin:0 auto; background:#000;
        border:1px solid rgba(255,215,0,0.25); box-shadow:0 6px 16px rgba(0,0,0,0.25);
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        will-change: transform, box-shadow;
      }
      .adv-poster img{
        width:100%; height:100%; object-fit:cover; display:block;
        transform-origin:center center; transition:transform .18s ease;
        will-change: transform;
      }
      .adv-card:hover .adv-poster{ transform:translateY(-3px); box-shadow:0 14px 32px rgba(0,0,0,0.35); border-color:rgba(255,215,0,0.45); }
      .adv-poster::after{ content:""; position:absolute; inset:0; background:linear-gradient(to top, rgba(0,0,0,.40), rgba(0,0,0,0) 55%); }

      /* Badge like */
      .mark-like{
        position:absolute; top:8px; left:8px; z-index:2; font-size:12px; font-weight:900;
        background:rgba(34,197,94,.16); color:#34d399; border:1px solid rgba(34,197,94,.45);
        padding:3px 8px; border-radius:10px;
      }

      /* Titre + méta + chips */
      .adv-title{
        margin:12px 0 4px 0; font-weight:900; font-size:21px; line-height:1.15;
        letter-spacing:-.2px; color:#fff;
        display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
      }
      .adv-meta{
        display:inline-flex; gap:12px; align-items:center; justify-content:flex-start;
        font-weight:900; font-size:14px; color:#fff; opacity:.97;
        padding:6px 10px; border-radius:10px; background:rgba(0,0,0,.28);
        border:1px solid rgba(255,215,0,.25);
      }
      .adv-chip{
        display:inline-block; margin:8px 0 4px 0; padding:4px 12px; border-radius:999px;
        background:rgba(255,215,0,0.12); color:#FFD700; font-size:13px; font-weight:800;
        border:1px solid rgba(255,215,0,0.35); max-width:var(--adv-w);
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
      }
      .chips-row{ display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 2px 0; }
      .chip-min{
        font-size:12px; padding:2px 8px; border-radius:999px;
        background:rgba(255,215,0,0.08); color:#FFD700; border:1px solid rgba(255,215,0,0.32); font-weight:800;
      }
      .adv-actors{ font-size:13.5px; opacity:.96; margin-top:6px; }
      .adv-desc{ color:rgba(255,255,255,0.94); font-size:15px; margin-top:8px; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }

      .cont-title{
        margin:20px 0 8px 0; text-transform:uppercase; font-weight:900; letter-spacing:.6px;
        border-left:3px solid #FFD700; padding-left:10px;
      }
      .adv-card [data-testid="column"] button, .adv-card .stButton > button{ width:100%; border-radius:999px; font-weight:800; }
    </style>
    """, unsafe_allow_html=True)

    # ============================
    #       CACHE DISQUE
    # ============================
    def _disk_get(key, ttl=900, path=".cache_reco.json"):
        try:
            if not os.path.exists(path):
                return None
            data = json.load(open(path, "r", encoding="utf-8"))
            it = data.get(key)
            if not it or (time.time() - it["ts"] > ttl):
                return None
            return it["val"]
        except Exception:
            return None

    def _disk_set(key, val, path=".cache_reco.json"):
        try:
            data = json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else {}
            data[key] = {"ts": time.time(), "val": val}
            json.dump(data, open(path, "w", encoding="utf-8"))
        except Exception:
            pass

    # ============================
    #         UTILITAIRES
    # ============================
    def minutes_to_hhmm(m):
        try:
            m = int(m or 0)
        except Exception:
            m = 0
        if m <= 0:
            return "Durée indisponible"
        h, mn = divmod(m, 60)
        return f"{h} h {mn:02d}"

    def _norm_genre_name(x: str) -> str:
        x = (x or "").strip().lower()
        rep = {
            "science fiction": "science-fiction",
            "sci-fi": "science-fiction",
            "comedie": "comédie",
            "animation": "animation",
            "thriller": "thriller",
            "romance": "romance",
            "drama": "drame",
            "horror": "horreur",
            "action": "action",
        }
        return rep.get(x, x)

    def _genres_to_set(gstr: str) -> set:
        parts = [p.strip() for p in (gstr or "").replace("/", ",").split(",") if p.strip()]
        return { _norm_genre_name(p) for p in parts }

    def split_genres(g):
        return [s for s in (g or "").replace("/", ",").split(",") if s.strip()]

    def synopsis_trim(txt, max_chars=220):
        """Tronque proprement au dernier séparateur avant max_chars."""
        if not txt:
            return None
        t = " ".join(str(txt).split())
        if len(t) <= max_chars:
            return t
        cut = max(
            t.rfind(". ", 0, max_chars),
            t.rfind("! ", 0, max_chars),
            t.rfind("? ", 0, max_chars),
            t.rfind(", ", 0, max_chars),
        )
        if cut == -1:
            cut = max_chars
        return t[:cut].rstrip(" ,;:")

    def brief_from_meta(title, genres, year, runtime_min):
        g = ", ".join([g for g in split_genres(genres)][:2]) or "Genre non précisé"
        bits = []
        if year: bits.append(str(year))
        if runtime_min: bits.append(minutes_to_hhmm(runtime_min))
        meta = " • ".join(bits)
        base = f"{title or 'Film'} — {g}"
        return f"{base}{(' • ' + meta) if meta else ''}."

    def smart_synopsis(title, overview, genres, data, max_chars=220, min_len=60):
        """
        Toujours retourner une phrase courte et lisible.
        - Si overview existe: on tronque proprement.
        - Si trop court ou vide: on génère un bref résumé à partir du titre + meta.
        """
        s = synopsis_trim(overview or "", max_chars=max_chars)
        if s and len(s) >= min_len:
            return s
        year = (data.get("release_date") or "")[:4]
        # résumé express basé sur le titre + meta (jamais vide)
        return brief_from_meta(title or data.get("name") or "Titre", genres, year, data.get("runtime") or 0)

    def synopsis_or_brief(ov, genres, data):
        # Conservée pour compatibilité, mais smart_synopsis est utilisée partout
        title = data.get("title") or data.get("name") or "Titre"
        return smart_synopsis(title, ov, genres, data, max_chars=220, min_len=60)

    @st.cache_data(show_spinner=False, ttl=86400)
    def get_age_certification(mid):
        try:
            d = tmdb_get(f"movie/{mid}", params={"append_to_response": "release_dates", "language": "fr-FR"}) or {}
            res = d.get("release_dates", {}).get("results", [])
            def pick(cc):
                for r in res:
                    if r.get("iso_3166_1") == cc:
                        for c in r.get("release_dates", []):
                            cert = (c.get("certification") or "").strip()
                            if cert: return cert
            return pick("FR") or pick("US") or "Tous publics"
        except Exception:
            return "Tous publics"

    @st.cache_data(show_spinner=False, ttl=3000)
    def fetch_top_cast_names(movie_id, limit=3):
        d = tmdb_get(f"movie/{movie_id}/credits", params={"language": "fr-FR"}) or {}
        cast = [c.get("name") for c in d.get("cast", []) if c.get("name")]
        return cast[:limit] if cast else []

    # >>> LIKES SYNC — helpers partagés avec la page "🎬 Recommandation"
    if "liked_movies" not in st.session_state:
        st.session_state["liked_movies"] = []

    def like_movie(mid: int):
        mid = int(mid)
        if mid not in st.session_state["liked_movies"]:
            st.session_state["liked_movies"].append(mid)

    def unlike_movie(mid: int):
        mid = int(mid)
        st.session_state["liked_movies"] = [m for m in st.session_state["liked_movies"] if m != mid]

    # =====================================
    #      SUGGESTIONS (si dataset local)
    # =====================================
    if HAS_MOVIES:
        # ---------- Corpus & TF-IDF ----------
        N_MAX = min(600, len(movies))

        @st.cache_data(show_spinner=False, ttl=3600)
        def build_corpus(df, n_max=600):
            use_cols = {
                "overview": next((c for c in ["overview_fr", "overview"] if c in df.columns), None),
                "genres":   next((c for c in ["genres_fr", "genres", "genre_names"] if c in df.columns), None),
                "title":    next((c for c in ["title_fr", "title", "name"] if c in df.columns), None),
                "id":       next((c for c in ["movie_id", "id"] if c in df.columns), None),
            }
            texts, mids, titles = [], [], []
            genres_by_mid = {}
            if use_cols["id"] is None:
                return texts, mids, titles, genres_by_mid

            st.session_state.setdefault("_adv_ov_cache", {})
            for _, r in df.head(n_max).iterrows():
                try:
                    mid = int(r[use_cols["id"]])
                except Exception:
                    continue

                ov_local = (r.get(use_cols["overview"]) if use_cols["overview"] else None)
                g_local  = (r.get(use_cols["genres"])   if use_cols["genres"]   else None)
                title    = (r.get(use_cols["title"])    if use_cols["title"]    else None)

                if (ov_local and str(ov_local).strip()) or (g_local and str(g_local).strip()):
                    ov, g = ov_local or "", g_local or ""
                else:
                    if mid in st.session_state._adv_ov_cache:
                        ov, g = st.session_state._adv_ov_cache[mid]
                    else:
                        ov, g, _ = fetch_overview_vote_genres_fast(mid)
                        st.session_state._adv_ov_cache[mid] = (ov or "", g or "")

                texts.append((f"{ov or ''} {g or ''}").strip() or (title or "Titre indisponible"))
                mids.append(mid)
                titles.append(title or "Titre indisponible")
                genres_by_mid[mid] = _genres_to_set(g)

            return texts, mids, titles, genres_by_mid

        @st.cache_data(show_spinner=False, ttl=3600)
        def compute_tfidf_sparse(texts):
            tfidf = TfidfVectorizer(max_features=2000, min_df=2, stop_words=None, dtype=np.float32)
            X  = tfidf.fit_transform(texts)
            Xn = _norm(X, norm="l2", axis=1, copy=False)
            return tfidf, Xn

        texts, mids, titles_dl, genres_by_mid = build_corpus(movies, N_MAX)
        tfidf, Xn = compute_tfidf_sparse(texts)
        D = Xn.shape[1]

        # ---------- Genres + feedback ----------
        ALL_GENRES = ["Action","Drame","Comédie","Romance","Science-Fiction","Thriller","Horreur","Animation"]
        DEFAULT_GENRES = ["Action","Drame"]
        st.session_state.setdefault("_adv_fav_genres", DEFAULT_GENRES.copy())
        if st.session_state.get("_adv_clear_genres", False):
            st.session_state._adv_clear_genres = False
            st.session_state._adv_fav_genres = []

        fav_genres = st.multiselect("🎭 Vos genres favoris :", ALL_GENRES, key="_adv_fav_genres")
        fav_set_norm = {_norm_genre_name(g) for g in fav_genres}

        def _build_reason_html(genres):
            if genres:
                chips = "".join([f"<span class='chip'>{html.escape(g)}</span>" for g in genres])
                lab = "le genre" if len(genres) == 1 else "les genres"
                return f"<div class='why-pill'>🎯 Car vous avez aimé {lab} : {chips}</div>"
            else:
                return "<div class='why-pill'>✨ Suggestions pour vous — basées sur votre activité récente.</div>"

        if st.session_state.get("_adv_reason_key") != tuple(st.session_state._adv_fav_genres):
            st.session_state._adv_reason_key  = tuple(st.session_state._adv_fav_genres)
            st.session_state._adv_reason_html = _build_reason_html(st.session_state._adv_fav_genres)

        st.markdown("<p class='section-lead'>🔥 Votre mix du moment — cliquez pour aimer/masquer et affiner le flux (sans bouger l’interface).</p>", unsafe_allow_html=True)
        if st.session_state.get("_adv_flash"):
            st.markdown(f"<div class='flash-pill'>✅ {html.escape(st.session_state['_adv_flash'])}</div>", unsafe_allow_html=True)
            st.session_state._adv_flash = ""
        st.markdown(st.session_state._adv_reason_html, unsafe_allow_html=True)

        # ---------- états Like/Dislike ----------
        st.session_state.setdefault("_adv_likes_mid", set())
        st.session_state.setdefault("_adv_dislikes_ids", set())
        st.session_state.setdefault("_adv_ui_disliked", set())

        # ---------- Profil & TOPK ----------
        st.session_state.setdefault("_adv_sorted_idx", [])
        st.session_state.setdefault("_adv_visible_idx", [])
        st.session_state.setdefault("_adv_cursor", 0)

        profile_key = tuple(sorted(fav_set_norm))

        # --------- filtre OR ----------
        def _genres_match(mid):
            gset = genres_by_mid.get(mid, set())
            if not fav_set_norm:
                return True
            return bool(gset & fav_set_norm)

        def _adv_refill_suggestions(replace_pos=None):
            lst = st.session_state._adv_sorted_idx
            vis = st.session_state._adv_visible_idx
            used = set(vis)
            dislikes = (st.session_state._adv_dislikes_ids | st.session_state._adv_ui_disliked)

            j = st.session_state._adv_cursor
            candidate = None

            while j < len(lst):
                i = lst[j]
                if (mids[i] not in dislikes) and (i not in used):
                    candidate = i
                    j += 1
                    break
                j += 1

            if candidate is None:
                j2 = 0
                while j2 < len(lst):
                    i = lst[j2]
                    if (mids[i] not in dislikes) and (i not in used):
                        candidate = i
                        j = j2 + 1
                        break
                    j2 += 1

            st.session_state._adv_cursor = j

            if candidate is None:
                return False
            if replace_pos is None or replace_pos >= len(vis):
                vis.append(candidate)
            else:
                vis[replace_pos] = candidate
            st.session_state._adv_visible_idx = vis
            return True

        def _build_adv_rankings():
            vg = (_norm(tfidf.transform([" ".join(st.session_state._adv_fav_genres)]),
                        norm="l2", axis=1, copy=False).toarray().ravel()
                 ) if st.session_state._adv_fav_genres else np.zeros(D, dtype=np.float32)
            scores = (Xn @ vg) if vg.any() else np.zeros(Xn.shape[0], dtype=np.float32)

            K = min(96, scores.size)
            idx_k = np.argpartition(-scores, K-1)[:K]
            idx_sorted = idx_k[np.argsort(-scores[idx_k])]

            seen = set()
            ordered = []
            for i in idx_sorted:
                mid = mids[i]
                if mid in seen:
                    continue
                seen.add(mid)
                ordered.append(i)

            if fav_set_norm:
                ordered = [i for i in ordered if _genres_match(mids[i])]
            if not ordered:
                ordered = list(idx_sorted)

            def coverage_weight(mid:int) -> int:
                gset = genres_by_mid.get(mid, set())
                return len(gset & fav_set_norm) if fav_set_norm else 0

            ordered.sort(key=lambda i: (float(scores[i]), coverage_weight(mids[i])), reverse=True)
            st.session_state._adv_sorted_idx = ordered
            st.session_state._adv_cursor = 0
            st.session_state._adv_visible_idx = []
            for _ in range(12):
                if not _adv_refill_suggestions():
                    break

        if st.session_state.get("_adv_last_key") != profile_key or not st.session_state._adv_sorted_idx:
            _build_adv_rankings()
            st.session_state._adv_last_key = profile_key
            st.session_state._adv_ui_disliked = set()
        else:
            vis = [i for i in st.session_state._adv_visible_idx
                   if mids[i] not in (st.session_state._adv_dislikes_ids | st.session_state._adv_ui_disliked)]
            st.session_state._adv_visible_idx = vis
            while len(st.session_state._adv_visible_idx) < 12 and _adv_refill_suggestions():
                pass

        # ---------- Cartes SUGGESTIONS ----------
        cols = st.columns(4)
        for slot, i in enumerate(st.session_state._adv_visible_idx[:12]):
            mid = mids[i]
            ov, genres_raw, data = fetch_overview_vote_genres_fast(mid)
            gset = genres_by_mid.get(mid, _genres_to_set(genres_raw))
            genres_text = ", ".join(sorted({g.title() for g in gset})) or (genres_raw or "Genres indisponibles")

            title  = (data.get("title") or data.get("name") or (titles_dl[i] if i < len(titles_dl) else None) or "Titre indisponible")

            # >>> IMAGES PLUS NETTES via srcset (w500 + w780) tout en affichant à 240px
            poster_500 = fetch_poster_fast(mid, size="w500")
            poster_780 = fetch_poster_fast(mid, size="w780") or poster_500
            img_tag = f"<img src='{poster_500}' srcset='{poster_500} 500w, {poster_780} 780w' sizes='(max-width: 260px) 240px, 240px' alt='{html.escape(title)}' loading='lazy'>"

            note   = float(data.get("vote_average") or 0)
            year   = (data.get("release_date") or "")[:4]
            runtime_hh = minutes_to_hhmm(data.get("runtime") or 0)
            age    = get_age_certification(mid)
            cast   = fetch_top_cast_names(mid, limit=3)

            # >>> SYNOPSIS TOUJOURS PROPRE
            desc = smart_synopsis(title, ov, genres_text, data, max_chars=220, min_len=60)

            card_id = f"card_{mid}"
            with cols[slot % 4]:
                st.markdown(f"<div id='{card_id}' class='adv-card'>", unsafe_allow_html=True)

                # Badge "Ajouté" si like actif
                if mid in st.session_state._adv_likes_mid:
                    st.markdown("<div class='mark-like'>❤️ Ajouté</div>", unsafe_allow_html=True)

                st.markdown(f"<div class='adv-poster'>{img_tag}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='adv-title'><b>{html.escape(title)}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='adv-chip'>{html.escape(genres_text)}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='adv-meta'>⭐ {note:.1f}/10 • 🕒 {runtime_hh} • 👤 {age} • 📅 {year or '—'}</div>", unsafe_allow_html=True)

                if cast:
                    st.markdown(f"<div class='adv-actors'><b>Avec :</b> {html.escape(', '.join(cast))}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='adv-desc'><em>{html.escape(desc)}</em></div>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("👍 J’aime", key=f"_adv_like_{mid}"):
                        # toggle + synchro globale
                        if mid in st.session_state._adv_likes_mid:
                            st.session_state._adv_likes_mid.discard(mid)
                            unlike_movie(mid)            # >>> LIKES SYNC
                            st.session_state._adv_flash = f"Retiré des préférences : {title}"
                        else:
                            st.session_state._adv_likes_mid.add(mid)
                            like_movie(mid)              # >>> LIKES SYNC
                            st.session_state._adv_flash = f"Ajouté aux préférences : {title}"
                        st.rerun()
                with c2:
                    if st.button("👎 Je n’aime pas", key=f"_adv_dislike_{mid}"):
                        st.session_state._adv_dislikes_ids.add(mid)
                        st.session_state._adv_ui_disliked.add(mid)
                        unlike_movie(mid)                # >>> LIKES SYNC (si était liké, on le retire)
                        try:
                            st.session_state._adv_visible_idx.pop(slot)
                        except Exception:
                            pass
                        _adv_refill_suggestions(replace_pos=slot)
                        st.markdown(f"<style>#{card_id}{{display:none!important}}</style>", unsafe_allow_html=True)
                        st.session_state._adv_flash = f"Masqué : {title}"
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

    # =====================================
    #            TENDANCES MONDE
    # =====================================
    show_trends = st.toggle("Afficher les tendances du moment", value=True)
    st.markdown("<p class='section-lead'>🌍 Ça bouge partout — affiches tendances par continent.</p>", unsafe_allow_html=True)

    AFR = {"ZA","DZ","MA","TN","EG","NG","KE","GH","SN","CI","ET","CM"}
    ASI = {"CN","HK","TW","JP","KR","IN","TH","VN","ID","PH","MY","SG","AE","SA"}
    EUR = {"FR","DE","ES","IT","GB","UK","IE","PT","NL","BE","CH","AT","SE","NO","DK","FI","PL","CZ","HU","RO","BG","GR","RU"}
    AME = {"US","CA","MX","BR","AR","CL","CO","PE","VE","UY","PR"}

    def _continent_from_codes(codes):
        s = set(codes)
        if s & AFR: return "Afrique"
        if s & ASI: return "Asie"
        if s & EUR: return "Europe"
        if s & AME: return "Amérique"
        return None

    @st.cache_data(show_spinner=False, ttl=900)
    def discover_by_countries(codes, limit=24):
        key = f"disc_{'-'.join(codes)}_{limit}"
        cached = _disk_get(key, ttl=900)
        if cached is not None:
            return cached
        try:
            d = tmdb_get("discover/movie", params={
                "with_origin_country":"|".join(codes),
                "sort_by":"popularity.desc","language":"fr-FR","page":1
            }) or {}
            res = d.get("results", [])[:limit]
            _disk_set(key, res)
            return res
        except Exception:
            return []

    @st.cache_data(show_spinner=False, ttl=900)
    def discover_by_countries_genre(codes, genre_id, limit=6):
        key = f"disc_{'-'.join(codes)}_g{genre_id}_{limit}"
        cached = _disk_get(key, ttl=900)
        if cached is not None:
            return cached
        try:
            d = tmdb_get("discover/movie", params={
                "with_origin_country":"|".join(codes),"with_genres":genre_id,
                "sort_by":"popularity.desc","language":"fr-FR","page":1
            }) or {}
            res = d.get("results", [])[:limit]
            _disk_set(key, res)
            return res
        except Exception:
            return []

    ASIA_EXTRA_TITLES = ["K-pop Demon Hunter","K-POP Demon Hunter","K-pop: Demon Hunter"]
    AFRICAN_TITLES    = ["Kirikou et la sorcière","Sarafina!","Jagun Jagun","Les Initiés",
                         "À la recherche du mari de ma femme","Le blanc d'Eyenga"]

    @st.cache_data(show_spinner=False, ttl=900)
    def search_titles_fr(titles):
        key = "search_" + "_".join(titles)
        cached = _disk_get(key, ttl=900)
        if cached is not None: 
            return cached
        found = []
        for q in titles:
            d = tmdb_get("search/movie", params={"query": q, "language": "fr-FR"}) or {}
            r = (d.get("results") or [])
            if not r: 
                continue
            m = r[0]; mid = m.get("id")
            if not mid: 
                continue
            ov, g, data = fetch_overview_vote_genres_fast(mid)
            # >>> images nettes pour tendances aussi
            poster_500 = fetch_poster_fast(mid, size="w500")
            year = (data.get("release_date") or "")[:4]
            title = data.get("title") or m.get("title") or q
            found.append({"id": mid, "title": title, "poster": poster_500, "genres": g, "overview": ov, "year": year})
        _disk_set(key, found)
        return found

    @st.cache_data(show_spinner=False, ttl=900)
    def tmdb_popular_fr(limit=150):
        key = f"popular_fr_{limit}"
        cached = _disk_get(key, ttl=900)
        if cached is not None:
            return cached
        items, page = [], 1
        while len(items) < limit and page <= 5:
            d = tmdb_get("movie/popular", params={"page": page, "language": "fr-FR"}) or {}
            items.extend(d.get("results", [])); page += 1
        items = items[:limit]; _disk_set(key, items); return items

    @st.cache_data(show_spinner=False, ttl=900)
    def build_trending_grouped(n_per=60, pool=200):
        hot = tmdb_popular_fr(limit=pool) or []
        out = {"Afrique":[], "Asie":[], "Europe":[], "Amérique":[]}
        seen_ids = set()

        def _push(mid, payload, continent):
            if mid in seen_ids or len(out[continent]) >= n_per: return
            seen_ids.add(mid); out[continent].append(payload)

        for m in hot:
            mid = m.get("id")
            if not mid: continue
            d = tmdb_movie_cached(mid) or {}
            pc = [c.get("iso_3166_1") for c in (d.get("production_countries") or []) if c.get("iso_3166_1")]
            cont = _continent_from_codes(pc)
            if not cont:
                lang = (d.get("original_language") or "").upper()
                if   lang in {"ZH","JA","KO","HI"}: cont="Asie"
                elif lang in {"FR","DE","ES","IT","RU","PT"}: cont="Europe"
                elif lang in {"EN","ES","PT"}: cont="Amérique"
                else: cont=None
            if not cont: continue
            ov, g, data = fetch_overview_vote_genres_fast(mid)
            poster_500 = fetch_poster_fast(mid, size="w500")
            year = (data.get("release_date") or "")[:4]
            title = data.get("title") or data.get("name") or m.get("title") or "Titre indisponible"
            _push(mid, {"id":mid,"title":title,"poster":poster_500,"genres":g,"overview":ov,"year":year}, cont)

        if len(out["Afrique"]) < n_per:
            extra = discover_by_countries(["NG","ZA","EG","MA","KE","GH","SN","CI","CM"], limit=(n_per-len(out["Afrique"])) * 2)
            for m in extra:
                mid = m.get("id")
                if not mid: continue
                ov, g, data = fetch_overview_vote_genres_fast(mid)
                poster_500 = fetch_poster_fast(mid, size="w500")
                year = (m.get("release_date") or data.get("release_date") or "")[:4]
                title = m.get("title") or data.get("title") or "Titre indisponible"
                _push(mid, {"id":mid,"title":title,"poster":poster_500,"genres":g,"overview":ov,"year":year}, "Afrique")
                if len(out["Afrique"]) >= n_per: break

        if len(out["Afrique"]) < n_per:
            for it in search_titles_fr(AFRICAN_TITLES):
                _push(it["id"], it, "Afrique")
                if len(out["Afrique"]) >= n_per: break

        for it in search_titles_fr(ASIA_EXTRA_TITLES):
            _push(it["id"], it, "Asie")

        GENRES_ENSURE = [("Action",28), ("Horreur",27), ("Animation",16)]
        CONTINENT_CODES = {
            "Afrique": ["NG","ZA","EG","MA","KE","GH","SN","CI","CM"],
            "Asie":    ["KR","JP","CN","IN","TH","VN","ID","PH","MY","TW","HK"],
            "Europe":  ["FR","DE","ES","IT","GB","UK","PT","NL","BE","SE","NO","DK","FI","PL","CZ"],
            "Amérique":["US","CA","BR","MX","AR","CL","CO","PE"]
        }
        for cont, codes in CONTINENT_CODES.items():
            have = " ".join(x.get("genres","") for x in out[cont]).lower()
            for lbl, gid in GENRES_ENSURE:
                if lbl.lower() in have: continue
                for m in discover_by_countries_genre(codes, gid, limit=3):
                    mid = m.get("id")
                    if not mid: continue
                    ov, g, data = fetch_overview_vote_genres_fast(mid)
                    poster_500 = fetch_poster_fast(mid, size="w500")
                    year = (m.get("release_date") or data.get("release_date") or "")[:4]
                    title = m.get("title") or data.get("title") or "Titre indisponible"
                    _push(mid, {"id":mid,"title":title,"poster":poster_500,"genres":g,"overview":ov,"year":year}, cont)
                    if len(out[cont]) >= n_per: break
        return out

    if show_trends:
        # --- bootstrap session_state ---
        st.session_state.setdefault("_adv_trends_grouped", {})
        st.session_state.setdefault("_adv_trends_visible", {})
        st.session_state.setdefault("_adv_trends_cursor", {})
        st.session_state.setdefault("_adv_trends_key", None)
        st.session_state.setdefault("_adv_dislikes_ids", st.session_state.get("_adv_dislikes_ids", set()))
        st.session_state.setdefault("_adv_ui_disliked", st.session_state.get("_adv_ui_disliked", set()))

        today = time.strftime("%Y-%m-%d")
        key = f"trends_{today}"
        DISPLAY_N = 9

        if st.session_state.get("_adv_trends_key") != key or not st.session_state._adv_trends_grouped:
            st.session_state._adv_trends_grouped = build_trending_grouped(n_per=60, pool=250)
            st.session_state._adv_trends_key = key
            st.session_state._adv_trends_visible = {c: [] for c in st.session_state._adv_trends_grouped.keys()}
            st.session_state._adv_trends_cursor  = {c: 0  for c in st.session_state._adv_trends_grouped.keys()}

        # fallback mini
        if not any(st.session_state._adv_trends_grouped.get(k) for k in ("Afrique","Asie","Europe","Amérique")):
            MIN_CODES = {
                "Afrique": ["NG","ZA","MA","EG","CM"],
                "Asie":    ["JP","KR","IN","CN","TH"],
                "Europe":  ["FR","DE","ES","IT","GB"],
                "Amérique":["US","CA","BR","MX","AR"]
            }
            for cont, codes in MIN_CODES.items():
                base = discover_by_countries(codes, limit=9) or []
                st.session_state._adv_trends_grouped[cont] = []
                for m in base[:9]:
                    mid = m.get("id")
                    if not mid:
                        continue
                    ov, g, data = fetch_overview_vote_genres_fast(mid)
                    poster_500 = fetch_poster_fast(mid, size="w500")
                    year = (m.get("release_date") or data.get("release_date") or "")[:4]
                    title = m.get("title") or data.get("title") or "Titre indisponible"
                    st.session_state._adv_trends_grouped[cont].append({
                        "id": mid, "title": title, "poster": poster_500,
                        "genres": g, "overview": ov, "year": year
                    })
                st.session_state._adv_trends_visible[cont] = list(range(min(9, len(st.session_state._adv_trends_grouped[cont]))))

        for c in ("Afrique","Asie","Europe","Amérique"):
            st.session_state._adv_trends_visible.setdefault(c, [])
            st.session_state._adv_trends_cursor.setdefault(c, 0)

        def _refill_one(continent, replace_pos=None):
            lst = st.session_state._adv_trends_grouped.get(continent, [])
            vis = st.session_state._adv_trends_visible.get(continent, [])
            used = set(vis)
            dislikes = st.session_state._adv_dislikes_ids | st.session_state._adv_ui_disliked

            j = st.session_state._adv_trends_cursor.get(continent, 0)
            candidate = None

            while j < len(lst):
                if lst[j]["id"] not in dislikes and j not in used:
                    candidate = j
                    j += 1
                    break
                j += 1

            if candidate is None:
                j2 = 0
                while j2 < len(lst):
                    if lst[j2]["id"] not in dislikes and j2 not in used:
                        candidate = j2
                        j = j2 + 1
                        break
                    j2 += 1

            st.session_state._adv_trends_cursor[continent] = j
            if candidate is None:
                return False

            if replace_pos is None or replace_pos >= len(vis):
                vis.append(candidate)
            else:
                vis[replace_pos] = candidate
            st.session_state._adv_trends_visible[continent] = vis
            return True

        # garantir DISPLAY_N
        for continent, lst in st.session_state._adv_trends_grouped.items():
            vis = [
                idx for idx in st.session_state._adv_trends_visible.get(continent, [])
                if 0 <= idx < len(lst)
                and lst[idx]["id"] not in st.session_state._adv_dislikes_ids
                and lst[idx]["id"] not in st.session_state._adv_ui_disliked
            ]
            st.session_state._adv_trends_visible[continent] = vis
            while len(st.session_state._adv_trends_visible[continent]) < DISPLAY_N:
                if not _refill_one(continent):
                    break

        # ----- rendu (sans boutons) -----
        for continent in ("Afrique","Asie","Europe","Amérique"):
            lst = st.session_state._adv_trends_grouped.get(continent, [])
            vis = st.session_state._adv_trends_visible.get(continent, [])[:DISPLAY_N]
            if not vis:
                continue

            st.markdown(f"<div class='cont-title'>| {continent.upper()} |</div>", unsafe_allow_html=True)
            cols = st.columns(3)

            for grid_pos, idx in enumerate(vis):
                if not (0 <= idx < len(lst)):
                    continue
                it = lst[idx]
                if it["id"] in st.session_state._adv_dislikes_ids or it["id"] in st.session_state._adv_ui_disliked:
                    continue

                # >>> synopsis garanti & propre
                syn = smart_synopsis(it.get("title"), it.get("overview"), it.get("genres"),
                                     {"release_date": f"{it.get('year','')}-01-01", "runtime": 0},
                                     max_chars=220, min_len=60)

                # >>> images nettes via srcset
                poster_500 = it.get("poster")
                poster_780 = poster_500  # si tu veux, tu peux générer w780 ici aussi
                img_tag = f"<img src='{poster_500}' srcset='{poster_500} 500w, {poster_780} 780w' sizes='(max-width: 260px) 240px, 240px' alt='{html.escape(it['title'])}' loading='lazy'>"

                with cols[grid_pos % 3]:
                    st.markdown("<div class='adv-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='adv-poster'>{img_tag}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='adv-title'><b>{html.escape(it['title'])}</b></div>", unsafe_allow_html=True)
                    chips = "".join(f"<span class='chip-min'>{html.escape(g)}</span>" for g in split_genres(it.get('genres', '')))
                    st.markdown(f"<div class='chips-row'>{chips}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='adv-meta'>📅 {it.get('year') or '—'}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='adv-desc'><em>{html.escape(syn)}</em></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # sécurité : stopper la page ici (évite que d'autres sections se superposent)
    st.stop()




#------- PARTIE 6--------------
# ─── PAGE CHATBOT (cinéma only, design + audio + salutations — SANS filtres UI) ───
elif page == "💬 FilmScope Chatbot":
    # --- Imports principaux de la page Chatbot ---
    import streamlit as st
    import os, base64, tempfile, re, html
    from gtts import gTTS                 # gTTS pour synthèse vocale (Google Text-to-Speech)
    from io import BytesIO                # buffer mémoire pour générer/servir l'audio sans fichier disque

    # --- Clés/API & options de génération ---
    GROQ_KEY   = os.getenv("KEY_GROQ") or os.getenv("KEY_GROQQ")  # clé API Groq si dispo (hébergé)
    USE_GROQ   = bool(GROQ_KEY)                                   # booléen : utiliser Groq si clé présente
    GROQ_MODEL = "llama-3.1-8b-instant"                           # modèle Groq par défaut
    TMDB_KEY   = os.getenv("TMDB_KEY")    # clé TMDB pour récupérer les affiches (recommandé)
    OMDB_KEY   = os.getenv("OMDB_KEY")    # clé OMDb en secours si TMDB absent

    # ============ STYLES ============
    # Look & feel du chatbot (bulles, avatars, textarea, etc.)
    st.markdown(
        """
<style>
.center{max-width:900px;margin:0 auto}
.fs-shell{padding:0;background:transparent;border:none;border-radius:0}

/* Titre au-dessus, affiche en dessous (grande) */
.title-block{display:flex;flex-direction:column;align-items:flex-start;gap:8px;margin:.2rem 0 .8rem}
.title-block .poster{width:min(260px,40vw);height:auto;border-radius:10px;border:1px solid #2a2a2a;object-fit:cover}
.title-block h1{margin:0 0 .25rem 0}

/* (styles prompt-avatars gardés mais non utilisés) */
.prompt-avatars{display:flex;align-items:center;gap:10px;margin:6px 0 0}
.prompt-avatars .pa-left{flex:1;display:flex;justify-content:flex-start}
.prompt-avatars .pa-right{flex:1;display:flex;justify-content:flex-end}
.prompt-avatars .avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#232323;border:1px solid #2a2a2a;font-size:20px}
.prompt-avatars .avatar.bot{color:#ffd166}
.prompt-avatars .avatar.user{color:#a0c4ff}

/* Texte d’accroche + input */
.fs-hook{color:#fff;font-size:1rem;margin:6px 0 8px 0;font-family:"Source Sans Pro",sans-serif}
.fs-input{position:relative}
.fs-input textarea{min-height:90px!important;height:90px!important;font-size:1rem!important;background:#1a1a1a;color:#eaeaea;border:1px solid #2a2a2a;border-radius:10px;font-family:"Source Sans Pro",sans-serif}

/* Bouton envoyer flottant (relook d'un bouton Streamlit) */
.fs-send{position:absolute;right:10px;bottom:10px;width:44px;height:44px;border-radius:10px;border:1px solid #2a2a2a;background:#1f1f1f;color:#fff}
.fs-send:hover{filter:brightness(1.1)}

/* Boutons généraux */
.btn{border-radius:10px;border:1px solid #2a2a2a;background:#1a1a1a;color:#eaeaea;padding:8px 14px;font-family:"Source Sans Pro",sans-serif;font-size:1rem}
.btn:hover{filter:brightness(1.08)}

/* Audio */
.fs-audio{margin-top:.6rem}
.fs-audio audio{width:100%}

/* Fil de messages + avatars */
.row{display:flex;gap:10px;margin:10px 0}
.row.user{justify-content:flex-end}   /* utilisateur à droite */
.row.bot{justify-content:flex-start}  /* bot à gauche */
.avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#232323;border:1px solid #2a2a2a}
.avatar.bot{color:#ffd166}
.avatar.user{color:#a0c4ff}

/* Bulles + étiquettes au-dessus */
.msg{display:flex;flex-direction:column;gap:6px;max-width:95%}
.msg-label{font-family:"Source Sans Pro",sans-serif;font-size:.75rem;line-height:1;padding:4px 6px;border-radius:4px;background:#2a2a2a;color:#eaeaea;opacity:.9}
.msg.bot .msg-label{align-self:flex-start;background:#5a3a2d} /* Réponse (bot) à gauche */
.msg.user .msg-label{align-self:flex-end;background:#2d3a5a}  /* Question (user) à droite */

/* Bulles de texte du chat */
.bubble{font-family:"Source Sans Pro",sans-serif;font-size:1.05rem;white-space:pre-wrap;word-wrap:break-word;overflow:visible;background:#141414;border:1px solid #2a2a2a;border-radius:6px;padding:10px 12px 12px}
.bubble--user{background:#151515}
.bubble--bot{background:#121212}
.bubble h1{font-size:1.35rem;margin:.35rem 0 .35rem;color:#fff;font-family:"Source Sans Pro",sans-serif}
.bubble h2{font-size:1.15rem;margin:.2rem 0 .25rem;color:#eaeaea;font-family:"Source Sans Pro",sans-serif}

/* Masquer la bande colorée Streamlit en haut */
header [data-testid="stDecoration"]{display:none!important}

/* === Affichage complet + cadre plus large (pas de coupe) === */
.center{max-width:1100px;}
.bubble{max-height:none!important;overflow:visible!important;word-break:break-word;}
</style>
""",
        unsafe_allow_html=True
    )

    # ============ ÉTAT ============
    # Variables d'état persistantes pour la session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []         # historique des messages (user/assistant)
    if "last_audio_b64" not in st.session_state:
        st.session_state.last_audio_b64 = None     # dernier audio généré en base64 (mémoire)
    if "last_assistant_text" not in st.session_state:
        st.session_state.last_assistant_text = ""  # dernier texte assistant (fallback TTS)
    if "_last_sent" not in st.session_state:       # anti double-submit
        st.session_state._last_sent = None
    if "_clear_input" not in st.session_state:     # vidage propre du textarea après reset
        st.session_state._clear_input = False
    if "_last_tts_text" not in st.session_state:   # éviter de régénérer l'audio si texte inchangé
        st.session_state._last_tts_text = ""

    # ── Sidebar : Historique (UN SEUL bloc Q/A)
    with st.sidebar:
        st.markdown("### 🗂️ Historique de la discussion")

        def _clean_preview(txt: str) -> str:
            """Nettoie le texte pour l'aperçu (supprime HTML/espaces)."""
            t = re.sub(r"<[^>]+>", "", txt or "")
            t = re.sub(r"\s+", " ", t).strip()
            return t

        if st.session_state.chat_history:
            pairs, pending_q = [], None
            for turn in st.session_state.chat_history:
                if turn.get("role") == "user":
                    pending_q = _clean_preview(turn.get("content", ""))
                elif turn.get("role") == "assistant" and pending_q is not None:
                    a = _clean_preview(turn.get("content", ""))
                    pairs.append((pending_q, a))
                    pending_q = None

            seen, compact = set(), []
            for q, a in pairs:
                key = (q, a)
                if compact and compact[-1] == key:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                compact.append(key)

            to_show = compact[-10:]
            for i, (q, a) in enumerate(to_show, start=1):
                st.markdown(f"**Q{i}.** {q}\n\n**A{i}.** {a}")
                if i < len(to_show):
                    st.markdown("---")
        else:
            st.info("Aucun message pour le moment.")

    # ============ Utils ============
    # Détection simple des salutations
    def is_greeting(txt: str) -> bool:
        t = txt.strip().lower()
        return any(re.search(p, t) for p in [
            r"^salut\b", r"^bonjour\b", r"^bonsoir\b", r"^coucou\b",
            r"^hey\b", r"^hello\b", r"^hi\b", r"^bjr\b", r"^slt\b"
        ])

    # Variantes "comment ça va ?"
    def is_how_are_you(txt: str) -> bool:
        t = txt.strip().lower()
        return any(re.search(p, t) for p in [
            r"\bcomment\s+vas[-\s]?tu\b",
            r"\bcomment\s+ça\s+va\b",
            r"\bca\s+va\b", r"\bça\s+va\b"
        ])

    # Réponse de salutation (avec extension "comment vas-tu")
    def greeting_reply(user_text: str | None = None) -> str:
        base = (
            "Bonjour et bienvenue. Je suis FilmScope Bot, votre assistant spécialisé dans le cinéma.\n\n"
            "En quoi puis-je vous aider ?\n"
            "- Fiche détaillée d’un film\n"
            "- Infos sur un acteur/actrice\n"
            "- Œuvres d’un réalisateur\n"
            "- Explorer un genre\n"
            "- Statistiques / box office\n"
            "- Où regarder en streaming"
        )
        if user_text and is_how_are_you(user_text):
            return "Je vais très bien, merci ! Et vous ? 🙂\n\n" + base
        return base

    # Détecte si la question parle de cinéma (FR/EN)
    def is_cinema_query(txt: str) -> bool:
        t = txt.lower()
        keywords = [
            # FR
            "film", "cinéma", "cinema", "acteur", "actrice", "réalisateur", "realisteur",
            "réalisatrice", "realisatrice", "casting", "distribution", "bande annonce",
            "trailer", "synopsis", "genre", "durée", "duree", "année", "annee", "sortie",
            "box-office", "box office", "affiche", "poster", "tournage", "scénario", "scenario",
            "oscar", "césar", "cesar", "palme", "festival", "imdb", "tmdb", "metacritic",
            "rottentomatoes", "netflix", "prime video", "amazon prime", "canal+", "canal plus",
            "ocs", "disney+", "hbo", "apple tv", "filmographie", "récompense", "recompense",
            "critique", "note", "étoiles", "etoiles",
            # EN
            "movie", "director", "actress", "actor", "screenplay", "runtime",
            "release year", "box office", "poster", "streaming", "where to watch",
            "award", "oscars", "golden globes"
        ]
        return any(k in t for k in keywords)

    # Conversion Markdown -> HTML (fallback regex si lib absente)
    def md_to_html(md_text: str) -> str:
        try:
            import markdown as mdlib
            return mdlib.markdown(md_text)
        except Exception:
            h = re.sub(r"^# (.+)$", r"<h1>\1</h1>", md_text, flags=re.M)
            h = re.sub(r"^## (.+)$", r"<h2>\1</h2>", h, flags=re.M)
            h = h.replace("\n\n", "<br><br>")
            return h

    # Extraction du titre (H1) et de l'année (depuis "## Année de parution") pour injecter l'affiche
    @st.cache_data(show_spinner=False, ttl=60*60*24)
    def _extract_title_and_year(md: str):
        title, year = None, None
        m = re.search(r'^\s*#\s*([^\n#]+)', md, flags=re.M)
        if m:
            title = m.group(1).strip()
        y = re.search(r'^\s*##\s*Année\s+de\s+parution\s*\n\s*(\d{4})', md, flags=re.M | re.I)
        if y:
            year = y.group(1)
        return title, year

    # Récupère l'URL d'affiche via TMDB (prioritaire) puis OMDb (fallback)
    @st.cache_data(show_spinner=False, ttl=60*60*24)
    def _fetch_poster_url(title: str, year: str | None = None) -> str | None:
        import requests, urllib.parse
        if TMDB_KEY:
            try:
                q = urllib.parse.quote(title)
                url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={q}"
                if year:
                    url += f"&year={year}"
                r = requests.get(url, timeout=12); r.raise_for_status()
                js = r.json()
                if js.get("results"):
                    poster_path = js["results"][0].get("poster_path")
                    if poster_path:
                        return f"https://image.tmdb.org/t/p/w500{poster_path}"
            except Exception:
                pass
        if OMDB_KEY:
            try:
                q = urllib.parse.quote(title)
                url = f"https://www.omdbapi.com/?apikey={OMDB_KEY}&t={q}"
                if year:
                    url += f"&y={year}"
                r = requests.get(url, timeout=12); r.raise_for_status()
                js = r.json()
                poster = js.get("Poster")
                if poster and poster != "N/A":
                    return poster
            except Exception:
                pass
        return None

    # Injecte l'affiche juste sous le H1 (mise en page verticale)
    def _inject_poster_next_to_h1(content_html: str, title: str, poster_url: str) -> str:
        m = re.search(r'<h1>(.*?)</h1>', content_html, flags=re.I | re.S)
        if not m:
            return content_html
        h1_inner = m.group(1)
        block = (
            f'<div class="title-block">'
            f'  <h1>{h1_inner}</h1>'
            f'  <img class="poster" src="{poster_url}" alt="Affiche : {html.escape(title)}" />'
            f'</div>'
        )
        return content_html.replace(m.group(0), block, 1)

    # ============ TITRE ============
    st.title("  🎬  FilmScope Chatbot")

    # ============ PROMPT centré + bouton ‘Envoyer 🎬’ ============
    # Message clair : cinéma uniquement + salutations autorisées
    st.markdown('<div class="center">', unsafe_allow_html=True)
    st.markdown(
        '<div class="fs-hook">Je réponds aux questions de <b>cinéma</b> uniquement. '
        'Hors cinéma, je ne réponds qu’aux salutations : <b>bonjour</b>, <b>bonsoir</b>, '
        '<b>salut</b>, <b>coucou</b>, <b>comment vas-tu</b> / <b>ça va</b>. '
        'Mes réponses sont <b>complètes</b> et <b>sans coupure</b>.</div>',
        unsafe_allow_html=True
    )

    # Vidage propre du champ après reset
    if "_clear_input" not in st.session_state:
        st.session_state._clear_input = False
    if st.session_state._clear_input:
        st.session_state["fs_input"] = ""
        st.session_state._clear_input = False

    # Formulaire principal (textarea + bouton envoyer relooké via JS/CSS)
    with st.container():
        st.markdown('<div class="fs-shell">', unsafe_allow_html=True)
        with st.form("fs_form", clear_on_submit=False):
            st.markdown('<div class="fs-input">', unsafe_allow_html=True)

            user_msg = st.text_area(
                label="",
                key="fs_input",
                height=90,
                placeholder="Posez votre question… (cinéma uniquement)",
                label_visibility="collapsed",
            )

            send_clicked = st.form_submit_button(" Envoyer🎬", help="Envoyer", type="secondary")
            st.markdown(
                """
<script>
const doc = window.parent.document;
const btns = Array.from(doc.querySelectorAll('button[kind="secondary"]'));
const sendBtn = btns.find(b => (b.textContent||'').toLowerCase().includes('envoyer'));
if (sendBtn) { sendBtn.classList.add('fs-send'); }
const ta = doc.querySelector('.fs-input textarea');
if (ta && sendBtn) {
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendBtn.click(); }
  });
}
</script>
""",
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)  # .fs-input

        left, right = st.columns([0.25, 0.25])
        with left:
            lire_clicked = st.button("▶️ Lire", key="fs_play", use_container_width=True, type="secondary")
        with right:
            clear_clicked = st.button("🧹 Effacer la discussion", key="fs_clear", use_container_width=True, type="secondary")
        st.markdown('</div>', unsafe_allow_html=True)  # .fs-shell
        st.markdown('</div>', unsafe_allow_html=True)  # .center

    # --- Effacement via bouton ---
    if clear_clicked:
        st.session_state.chat_history = []
        st.session_state.last_audio_b64 = None
        st.session_state.last_assistant_text = ""
        st.session_state._last_sent = None
        st.session_state._clear_input = True
        st.rerun()

    # ============ GÉNÉRATION : CINÉMA UNIQUEMENT (pas de filtres UI) ============
    # Prompt système : CINÉMA only + réponses entières
    system_prefix = (
        "Tu es **FilmScope Bot**, expert du cinéma. "
        "🚫 Tu ne réponds **qu’aux questions liées au cinéma** (films, acteurs/actrices, réalisateurs/trices, genres, synopsis, box-office, tournage, plateformes, distinctions, etc.). "
        "Si la question n’est **pas** liée au cinéma, **ne réponds pas au fond** : indique poliment que tu es spécialisé en cinéma et invite à reformuler. "
        "✅ Exception : tu peux répondre aux **salutations** (bonjour, bonsoir, salut, coucou, comment vas-tu / ça va).\n\n"
        "Réponds en **Markdown** clair, **structuré** et **en entier** (ne tronque jamais). "
        "Pour la fiche d’un film, respecte EXACTEMENT le format :\n"
        "# {Titre du film}\n"
        "*Contexte du titre* : 3 phrases précises.\n"
        "## Acteurs\n- 4 à 8 principaux.\n"
        "## Duree\n- heure du films.\n"
        "## Genre\n- 1 à 3 genres.\n"
        "## Synopsis\n2 à 5 phrases complètes.\n"
        "## Année de parution\nYYYY\n** : explique en 4 phrases complètes.\n"
        "## Lieux de tournage (principaux)\n- 2 à 4 lieux.\n** : plusieurs phrases complètes sur le choix et l’influence de ces lieux."
    )
    MAX_TOKENS = 1024  # taille fixe (aucun slider)

    # Appel Groq hébergé (si clé dispo)
    def generate_remote_groq(system_prompt: str, user_prompt: str) -> str:
        import requests
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_KEY or ''}"}
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": int(MAX_TOKENS),
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # Modèle local minimal (fallback)
    @st.cache_resource(show_spinner=False)
    def load_tiny_local():
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True, use_safetensors=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        mdl = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype="auto", low_cpu_mem_usage=True, device_map="cpu", use_safetensors=True
        )
        mdl.eval()
        return tok, mdl

    def generate_local(system_prompt: str, user_prompt: str) -> str:
        import torch
        tok, mdl = load_tiny_local()
        full = f"{system_prompt}\n\n{user_prompt}"
        with torch.inference_mode():
            out = mdl.generate(
                **tok(full, return_tensors="pt"),
                max_new_tokens=int(MAX_TOKENS),
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tok.pad_token_id or tok.eos_token_id
            )
        return tok.batch_decode(out, skip_special_tokens=True)[0].replace(full, "").strip()

    # —— Envoi : salutations OK ; sinon CINÉMA obligatoire ; sinon message de redirection
    if send_clicked and (user_msg or "").strip():
        um = user_msg.strip()
        if um != (st.session_state.get("_last_sent") or ""):
            st.session_state._last_sent = um

            if is_greeting(um) or is_how_are_you(um):
                # Salutations autorisées
                bot_response = greeting_reply(um)

            elif not is_cinema_query(um):
                # Hors cinéma → message dédié
                bot_response = (
                    "Je suis spécialisé dans le **cinéma** 🎬 et je ne traite pas les questions hors de ce domaine.\n"
                    "Tu peux me demander, par exemple :\n"
                    "- la fiche d’un film (acteurs, genre, durée, synopsis, année de parution) ;\n"
                    "- des infos sur un(e) acteur/actrice ou un(e) réalisateur(trice) ;\n"
                    "- le box-office, des distinctions, ou où regarder un film en streaming.\n"
                    "J’accepte aussi les salutations : **bonjour**, **bonsoir**, **salut**, **coucou**, **comment vas-tu / ça va**."
                )
            else:
                # Demande cinéma → génération complète
                try:
                    bot_response = generate_remote_groq(system_prefix, um) if USE_GROQ else generate_local(system_prefix, um)
                except Exception:
                    bot_response = generate_local(system_prefix, um)

            # Historique + TTS (l’audio lira la DERNIÈRE réponse du bot)
            st.session_state.chat_history.append({"role": "user", "content": html.escape(um)})
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            st.session_state.last_assistant_text = bot_response

    # ============ Rendu fil (Markdown → HTML) ============
    def _render_message(role: str, content_md: str):
        content_html = md_to_html(content_md)
        left_avatar  = '<div class="avatar bot">🎬</div>'
        right_avatar = '<div class="avatar user">🙂</div>'

        if role == "assistant":
            try:
                title, year = _extract_title_and_year(content_md)
                if title:
                    poster_url = _fetch_poster_url(title, year)
                    if poster_url:
                        content_html = _inject_poster_next_to_h1(content_html, title, poster_url)
            except Exception:
                pass

            html_msg = (
                f'<div class="row bot">{left_avatar}'
                f'  <div class="msg bot">'
                f'    <div class="msg-label">Réponse</div>'
                f'    <div class="bubble bubble--bot">{content_html}</div>'
                f'  </div>'
                f'</div>'
            )
        else:
            html_msg = (
                f'<div class="row user">'
                f'  <div class="msg user">'
                f'    <div class="msg-label">Question</div>'
                f'    <div class="bubble bubble--user">{content_html}</div>'
                f'  </div>'
                f'{right_avatar}</div>'
            )
        st.markdown(html_msg, unsafe_allow_html=True)

    for turn in st.session_state.chat_history:
        _render_message(turn["role"], turn["content"])

    # ============ Lire → gTTS (dernier texte du BOT) ============
    def _get_last_assistant_text() -> str:
        for turn in reversed(st.session_state.get("chat_history", [])):
            if turn.get("role") == "assistant":
                return (turn.get("content") or "").strip()
        return (st.session_state.get("last_assistant_text") or "").strip()

    def _sanitize_for_tts(text: str) -> str:
        """Nettoie Markdown/HTML/balises/symboles pour TTS (lettres/chiffres/espaces)."""
        import unicodedata
        t = re.sub(r"```.*?```", " ", text, flags=re.S)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"\[(.*?)\]\([^)]+\)", r"\1", t)
        t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)
        t = re.sub(r"^\s*[-*•]\s*", "", t, flags=re.M)
        t = re.sub(r"\*\*([^*]+)\*\*|\*([^*]+)\*", r"\1\2", t)
        t = unicodedata.normalize("NFKC", t)
        t = "".join(ch if (unicodedata.category(ch)[0] in ("L", "N") or ch.isspace()) else " " for ch in t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    audio_id = "filmscope_player"
    if lire_clicked:
        raw_bot_text   = _get_last_assistant_text()
        clean_bot_text = _sanitize_for_tts(raw_bot_text)

        if not clean_bot_text:
            st.info("Aucune réponse du bot à lire pour l’instant 🙂")
        else:
            try:
                if clean_bot_text != st.session_state.get("_last_tts_text", ""):
                    tts = gTTS(text=clean_bot_text, lang="fr", slow=False)
                    buf = BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    st.session_state.last_audio_b64 = base64.b64encode(buf.read()).decode()
                    st.session_state._last_tts_text = clean_bot_text
            except Exception as e:
                st.error(f"Échec de la génération audio : {e}")

    if st.session_state.last_audio_b64:
        st.markdown('<div class="fs-audio">', unsafe_allow_html=True)
        st.markdown(
            f'<audio id="{audio_id}" preload="metadata" controls src="data:audio/mpeg;base64,{st.session_state.last_audio_b64}"></audio>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        if lire_clicked:
            st.components.v1.html(
                f"<script>var a=document.getElementById('{audio_id}');if(a){{a.pause();a.currentTime=0;a.load();a.addEventListener('canplaythrough',function h(){{a.removeEventListener('canplaythrough',h);a.play();}});}}</script>",
                height=0
            )
