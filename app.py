"""
AirRoute Intelligence — Professional Streamlit Dashboard
Temporal Link Prediction on tgbl-flight (TGB Benchmark)
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page config & paths
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AirRoute Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(RESULTS, "figures")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #2196F3;
    }
    div[data-testid="stMetric"] label { font-size: 0.85rem; color: #555; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_dashboard_data():
    return load_json("dashboard_data.json")


@st.cache_data
def get_airport_list():
    """Return sorted list of airports that have stats."""
    dd = load_dashboard_data()
    if not dd:
        return []
    stats = dd.get("airport_stats", {})
    meta = dd.get("airport_meta", {})
    items = []
    for code, s in stats.items():
        m = meta.get(code, {})
        name = m.get("name", code)
        flights = s.get("flights_total", 0)
        items.append((code, name, flights))
    items.sort(key=lambda x: -x[2])
    return items


# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
COLORS = {
    "Random": "#9E9E9E",
    "HistFreq": "#2196F3",
    "Recency": "#FF9800",
    "EdgeBank": "#4CAF50",
    "TGN": "#E91E63",
}
MODELS = ["Random", "HistFreq", "Recency", "EdgeBank", "TGN"]
MODEL_TYPES = {
    "Random": "Heuristique",
    "HistFreq": "Heuristique",
    "Recency": "Heuristique",
    "EdgeBank": "Heuristique",
    "TGN": "Neural (TGN)",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ✈️ AirRoute Intelligence")
st.sidebar.caption("Temporal Link Prediction on **tgbl-flight**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🌍 Overview",
        "🔎 Airport Explorer",
        "🔮 Future Link Predictions",
        "🛡️ Network Resilience Simulator",
        "📊 Model Performance",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset**: tgbl-flight (TGB)\n\n"
    "**Benchmark**: NeurIPS 2023\n\n"
    "**Periode**: 2019–2022"
)


# ===================================================================
# PAGE 1: OVERVIEW
# ===================================================================
def page_overview():
    st.markdown("# 🌍 Overview")
    st.markdown("*Tableau de bord global du reseau aerien mondial*")

    # KPIs
    dd = load_dashboard_data()
    resilience = load_json("resilience_results.json")
    test = load_json("test_results.json")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Aeroports", "18,143")
    c2.metric("Interactions", "67.2M")
    if dd:
        gs = dd["graph_summary"]
        c3.metric("Routes actives", f"{gs['num_routes']:,}")
        c4.metric("Vols (3 mois)", f"{gs['total_flights']:,}")
    else:
        c3.metric("Routes actives", "505,101")
        c4.metric("Vols (3 mois)", "4,846,953")
    if test:
        c5.metric("Meilleur MRR", f"{test['HistFreq']['mrr']:.3f}")

    st.markdown("---")

    # World map
    st.markdown("### 🗺️ Carte du reseau aerien mondial")
    if dd and dd.get("map_edges"):
        with st.spinner("Construction de la carte..."):
            fig = _build_world_map(dd)
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Routes avec 50+ vols sur 3 mois (aout-nov 2021). "
            f"{len(dd['map_edges']):,} routes affichees."
        )
    else:
        st.warning("Donnees cartographiques non disponibles. "
                   "Executez `python scripts/prepare_dashboard_data.py`.")

    st.markdown("---")

    # Key results summary
    st.markdown("### 📈 Resultats cles")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Performance des modeles (Test)")
        if test:
            fig = go.Figure()
            mrr_vals = [test[m]["mrr"] for m in MODELS]
            fig.add_trace(go.Bar(
                x=MODELS, y=mrr_vals,
                marker_color=[COLORS[m] for m in MODELS],
                text=[f"{v:.3f}" for v in mrr_vals],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis_title="MRR", yaxis_range=[0, 1],
                height=350, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Resilience du reseau")
        if resilience:
            targeted = resilience["disruption"]["targeted"]
            random_ = resilience["disruption"]["random"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[r["removed"] for r in targeted],
                y=[r["connectivity"] * 100 for r in targeted],
                mode="lines", name="Attaque ciblee",
                line=dict(color="#E91E63", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=[r["removed"] for r in random_],
                y=[r["connectivity"] * 100 for r in random_],
                mode="lines", name="Panne aleatoire",
                line=dict(color="#2196F3", width=2),
            ))
            fig.update_layout(
                xaxis_title="Aeroports retires",
                yaxis_title="Connectivite (%)",
                yaxis_range=[80, 90],
                height=350, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


def _build_world_map(dd):
    """Build a Plotly world map with airport nodes and route edges."""
    meta = dd["airport_meta"]
    map_edges = dd["map_edges"]
    stats = dd.get("airport_stats", {})

    # Route lines
    lons, lats = [], []
    for e in map_edges:
        lats += [e["src_lat"], e["dst_lat"], None]
        lons += [e["src_lon"], e["dst_lon"], None]

    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        mode="lines",
        line=dict(width=0.3, color="rgba(65, 105, 225, 0.15)"),
        hoverinfo="skip",
        name="Routes",
    ))

    # Airport dots (only those with coordinates and some traffic)
    ap_lats, ap_lons, ap_texts, ap_sizes, ap_colors = [], [], [], [], []
    for code, m in meta.items():
        if m.get("lat") is None:
            continue
        s = stats.get(code, {})
        flights = s.get("flights_total", 0)
        if flights < 10:
            continue
        ap_lats.append(m["lat"])
        ap_lons.append(m["lon"])
        ap_texts.append(
            f"<b>{code}</b><br>{m.get('name', '')}<br>"
            f"{m.get('city', '')}, {m.get('country', '')}<br>"
            f"Vols: {flights:,}"
        )
        sz = max(2, min(15, flights / 5000))
        ap_sizes.append(sz)
        ap_colors.append(flights)

    fig.add_trace(go.Scattergeo(
        lat=ap_lats, lon=ap_lons,
        mode="markers",
        marker=dict(
            size=ap_sizes,
            color=ap_colors,
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="Vols", x=1.02),
            line=dict(width=0.3, color="white"),
        ),
        text=ap_texts,
        hoverinfo="text",
        name="Aeroports",
    ))

    fig.update_geos(
        showcountries=True, countrycolor="lightgray",
        showcoastlines=True, coastlinecolor="lightgray",
        showland=True, landcolor="#f0f0f0",
        showocean=True, oceancolor="#e8f4fd",
        showlakes=False,
        projection_type="natural earth",
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig


# ===================================================================
# PAGE 2: AIRPORT EXPLORER
# ===================================================================
def page_airport_explorer():
    st.markdown("# 🔎 Airport Explorer")
    st.markdown("*Explorez les connexions, centralites et communautes de chaque aeroport*")

    dd = load_dashboard_data()
    if not dd:
        st.error("Donnees non disponibles. Executez `python scripts/prepare_dashboard_data.py`.")
        return

    meta = dd["airport_meta"]
    stats = dd["airport_stats"]
    resilience = load_json("resilience_results.json")

    # Airport selector
    airports = get_airport_list()
    options = [f"{code} — {name}" for code, name, _ in airports[:500]]

    selected = st.selectbox(
        "Selectionner un aeroport",
        options,
        index=0,
        help="Les 500 aeroports avec le plus de trafic",
    )
    code = selected.split(" — ")[0]

    m = meta.get(code, {})
    s = stats.get(code, {})

    if not s:
        st.warning(f"Aucune donnee disponible pour {code}.")
        return

    # Header
    st.markdown(f"## {code} — {m.get('name', 'N/A')}")
    st.markdown(f"📍 {m.get('city', '')}, {m.get('country', '')}")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vols totaux", f"{s['flights_total']:,}")
    c2.metric("Departs", f"{s['flights_out']:,}")
    c3.metric("Arrivees", f"{s['flights_in']:,}")
    c4.metric("Connexions (in+out)", f"{s['degree_total']:,}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["Connexions", "Centralite", "Carte"]
    )

    with tab1:
        st.markdown("### Top destinations depuis " + code)
        conns = s.get("top_connections", [])
        if conns:
            rows = []
            for c_item in conns:
                dst = c_item["dst"]
                dst_meta = meta.get(dst, {})
                rows.append({
                    "Destination": dst,
                    "Nom": dst_meta.get("name", "N/A"),
                    "Ville": dst_meta.get("city", ""),
                    "Pays": dst_meta.get("country", ""),
                    "Vols": c_item["flights"],
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Bar chart
            fig = go.Figure(go.Bar(
                x=[r["Destination"] for r in rows[:15]],
                y=[r["Vols"] for r in rows[:15]],
                marker_color="#2196F3",
                text=[f"{r['Vols']:,}" for r in rows[:15]],
                textposition="outside",
            ))
            fig.update_layout(
                xaxis_title="Destination",
                yaxis_title="Nombre de vols",
                height=350, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune connexion sortante enregistree.")

    with tab2:
        st.markdown("### Position dans les classements de centralite")
        if resilience:
            top = resilience["top_airports"]
            metrics_data = {
                "Degre pondere": top["by_degree"],
                "Betweenness": top["by_betweenness"],
                "PageRank": top["by_pagerank"],
            }
            rows = []
            for metric_name, ranking in metrics_data.items():
                if code in ranking:
                    rank = ranking.index(code) + 1
                    rows.append({
                        "Metrique": metric_name,
                        "Rang": f"#{rank}",
                        "Statut": "🏆 Top 20" if rank <= 20 else "",
                    })
                else:
                    rows.append({
                        "Metrique": metric_name,
                        "Rang": "> #20",
                        "Statut": "",
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Show where this airport sits in the top lists
            for metric_name, ranking in metrics_data.items():
                if code in ranking[:10]:
                    rank = ranking.index(code) + 1
                    st.success(f"**{metric_name}**: {code} est #{rank} sur 13,986 aeroports")
        else:
            st.info("Resultats de resilience non disponibles.")

    with tab3:
        st.markdown(f"### Connexions de {code} sur la carte")
        if m.get("lat") is not None and conns:
            fig = go.Figure()

            # Route lines from this airport
            for c_item in conns[:20]:
                dst = c_item["dst"]
                dst_meta = meta.get(dst, {})
                if dst_meta.get("lat") is None:
                    continue
                opacity = min(1.0, c_item["flights"] / conns[0]["flights"])
                fig.add_trace(go.Scattergeo(
                    lat=[m["lat"], dst_meta["lat"]],
                    lon=[m["lon"], dst_meta["lon"]],
                    mode="lines",
                    line=dict(
                        width=max(1, 4 * opacity),
                        color=f"rgba(33, 150, 243, {max(0.3, opacity)})",
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                ))

            # Destination dots
            dst_lats, dst_lons, dst_texts = [], [], []
            for c_item in conns[:20]:
                dst = c_item["dst"]
                dst_meta = meta.get(dst, {})
                if dst_meta.get("lat") is None:
                    continue
                dst_lats.append(dst_meta["lat"])
                dst_lons.append(dst_meta["lon"])
                dst_texts.append(
                    f"<b>{dst}</b><br>{dst_meta.get('name', '')}<br>"
                    f"Vols: {c_item['flights']:,}"
                )

            fig.add_trace(go.Scattergeo(
                lat=dst_lats, lon=dst_lons,
                mode="markers",
                marker=dict(size=7, color="#2196F3", line=dict(width=1, color="white")),
                text=dst_texts, hoverinfo="text",
                name="Destinations",
            ))

            # Source airport
            fig.add_trace(go.Scattergeo(
                lat=[m["lat"]], lon=[m["lon"]],
                mode="markers",
                marker=dict(size=14, color="#E91E63", symbol="star",
                            line=dict(width=1, color="white")),
                text=[f"<b>{code}</b><br>{m.get('name', '')}"],
                hoverinfo="text",
                name=code,
            ))

            fig.update_geos(
                showcountries=True, countrycolor="lightgray",
                showcoastlines=True, coastlinecolor="lightgray",
                showland=True, landcolor="#f0f0f0",
                showocean=True, oceancolor="#e8f4fd",
                projection_type="natural earth",
            )
            fig.update_layout(
                height=450, margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Coordonnees non disponibles pour cet aeroport.")


# ===================================================================
# PAGE 3: FUTURE LINK PREDICTIONS
# ===================================================================
def page_predictions():
    st.markdown("# 🔮 Future Link Predictions")
    st.markdown("*Predire les prochaines connexions d'un aeroport*")

    dd = load_dashboard_data()
    if not dd:
        st.error("Donnees non disponibles. Executez `python scripts/prepare_dashboard_data.py`.")
        return

    meta = dd["airport_meta"]
    predictions = dd.get("predictions", {})

    st.info(
        "**Modele utilise**: HistFreq (frequence historique) — notre meilleur modele "
        "sur la metrique MRR (0.738 sur le set de test). Les scores representent "
        "la probabilite relative basee sur la frequence observee des routes.\n\n"
        "Le modele TGN (0.653 MRR) necessite un warmup memoire de ~50 minutes sur CPU "
        "et n'est pas disponible en temps reel."
    )

    # Airport selector
    airports = get_airport_list()
    options = [f"{code} — {name}" for code, name, _ in airports[:500]]

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox(
            "Aeroport source",
            options,
            key="pred_airport",
        )
    with col2:
        top_k = st.slider("Nombre de predictions", 5, 20, 10)

    code = selected.split(" — ")[0]
    preds = predictions.get(code, [])

    if not preds:
        st.warning(f"Aucune prediction disponible pour {code}.")
        return

    m = meta.get(code, {})
    st.markdown(f"### Destinations predites depuis **{code}** ({m.get('name', '')})")

    preds_shown = preds[:top_k]

    # Results table
    rows = []
    for i, p in enumerate(preds_shown):
        dst = p["dst"]
        dst_meta = meta.get(dst, {})
        rows.append({
            "Rang": i + 1,
            "Destination": dst,
            "Nom": dst_meta.get("name", "N/A"),
            "Ville": dst_meta.get("city", ""),
            "Pays": dst_meta.get("country", ""),
            "Score": p["score"],
            "Vols historiques": p["flights"],
        })

    df = pd.DataFrame(rows)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.dataframe(
            df.style.format({"Score": "{:.4f}", "Vols historiques": "{:,}"}),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        # Score bar chart
        fig = go.Figure(go.Bar(
            y=[r["Destination"] for r in rows[::-1]],
            x=[r["Score"] for r in rows[::-1]],
            orientation="h",
            marker_color=["#E91E63" if i == len(rows) - 1 else "#64B5F6"
                          for i in range(len(rows))],
            text=[f"{r['Score']:.3f}" for r in rows[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Score de prediction (probabilite relative)",
            xaxis_title="Score",
            height=max(300, 35 * len(rows)),
            margin=dict(l=80, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Map of predicted connections
    st.markdown("### 🗺️ Carte des destinations predites")
    if m.get("lat") is not None:
        fig = go.Figure()

        for p in preds_shown:
            dst = p["dst"]
            dst_meta = meta.get(dst, {})
            if dst_meta.get("lat") is None:
                continue
            fig.add_trace(go.Scattergeo(
                lat=[m["lat"], dst_meta["lat"]],
                lon=[m["lon"], dst_meta["lon"]],
                mode="lines",
                line=dict(width=max(1, 4 * p["score"] / preds_shown[0]["score"]),
                          color="rgba(233, 30, 99, 0.5)"),
                hoverinfo="skip", showlegend=False,
            ))

        # Destination markers
        dst_lats, dst_lons, dst_texts, dst_sizes = [], [], [], []
        for p in preds_shown:
            dst = p["dst"]
            dst_meta = meta.get(dst, {})
            if dst_meta.get("lat") is None:
                continue
            dst_lats.append(dst_meta["lat"])
            dst_lons.append(dst_meta["lon"])
            dst_texts.append(
                f"<b>{dst}</b><br>{dst_meta.get('name', '')}<br>"
                f"Score: {p['score']:.4f}<br>Vols: {p['flights']:,}"
            )
            dst_sizes.append(max(6, 14 * p["score"] / preds_shown[0]["score"]))

        fig.add_trace(go.Scattergeo(
            lat=dst_lats, lon=dst_lons,
            mode="markers+text",
            marker=dict(size=dst_sizes, color="#2196F3",
                        line=dict(width=1, color="white")),
            text=[p["dst"] for p in preds_shown if meta.get(p["dst"], {}).get("lat")],
            textposition="top center", textfont=dict(size=9),
            hovertext=dst_texts, hoverinfo="text",
            name="Predictions",
        ))

        # Source
        fig.add_trace(go.Scattergeo(
            lat=[m["lat"]], lon=[m["lon"]],
            mode="markers",
            marker=dict(size=16, color="#E91E63", symbol="star",
                        line=dict(width=1, color="white")),
            text=[f"<b>{code}</b><br>{m.get('name', '')}"],
            hoverinfo="text", name="Source",
        ))

        fig.update_geos(
            showcountries=True, countrycolor="lightgray",
            showcoastlines=True, coastlinecolor="lightgray",
            showland=True, landcolor="#f0f0f0",
            showocean=True, oceancolor="#e8f4fd",
            projection_type="natural earth",
        )
        fig.update_layout(
            height=450, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE 4: NETWORK RESILIENCE SIMULATOR
# ===================================================================
def page_resilience_simulator():
    st.markdown("# 🛡️ Network Resilience Simulator")
    st.markdown("*Simulez la suppression d'un aeroport et observez l'impact sur le reseau*")

    resilience = load_json("resilience_results.json")
    dd = load_dashboard_data()

    if not resilience or not dd:
        st.error("Donnees non disponibles.")
        return

    meta = dd["airport_meta"]
    stats = dd["airport_stats"]
    gs = resilience["graph_stats"]

    # Network baseline metrics
    st.markdown("### Etat initial du reseau")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aeroports", f"{gs['nodes']:,}")
    c2.metric("Routes", f"{gs['edges']:,}")
    baseline = resilience["disruption"]["targeted"][0]
    c3.metric("Plus grande SCC", f"{baseline['largest_scc']:,}")
    c4.metric("Connectivite", f"{baseline['connectivity']*100:.1f}%")

    st.markdown("---")

    tab1, tab2 = st.tabs(
        ["Simulateur interactif", "Simulation pre-calculee"]
    )

    with tab1:
        st.markdown("### Selectionner un aeroport a supprimer")
        st.markdown(
            "Choisissez un aeroport et visualisez son importance pour le reseau."
        )

        # Show top critical airports
        top_bc = resilience["top_airports"]["by_betweenness"][:20]
        top_deg = resilience["top_airports"]["by_degree"][:20]

        col1, col2 = st.columns([2, 1])
        with col1:
            airport_options = []
            for code in top_bc[:20]:
                m = meta.get(code, {})
                airport_options.append(
                    f"{code} — {m.get('name', 'N/A')} (betweenness top)"
                )
            # Add top degree airports not already in list
            for code in top_deg[:20]:
                if code not in top_bc[:20]:
                    m = meta.get(code, {})
                    airport_options.append(
                        f"{code} — {m.get('name', 'N/A')} (degree top)"
                    )

            selected = st.selectbox(
                "Aeroport a supprimer",
                airport_options,
                key="resilience_airport",
            )

        with col2:
            st.markdown("**Classement de criticite**")
            selected_code = selected.split(" — ")[0]
            for metric, ranking in [
                ("Betweenness", top_bc),
                ("Degre", top_deg),
            ]:
                if selected_code in ranking:
                    rank = ranking.index(selected_code) + 1
                    st.markdown(f"- {metric}: **#{rank}**")
                else:
                    st.markdown(f"- {metric}: > #20")

        # Impact analysis
        s = stats.get(selected_code, {})
        m = meta.get(selected_code, {})

        st.markdown(f"### Impact de la suppression de **{selected_code}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Vols affectes", f"{s.get('flights_total', 0):,}")
        c2.metric("Routes supprimees", f"{s.get('degree_total', 0):,}")
        c3.metric(
            "% du trafic total",
            f"{s.get('flights_total', 0) / gs['total_flights'] * 100:.2f}%",
        )

        # Show what this airport connects
        conns = s.get("top_connections", [])
        if conns:
            st.markdown("**Connexions perdues (top 10):**")
            conn_rows = []
            for c_item in conns[:10]:
                dst_meta = meta.get(c_item["dst"], {})
                conn_rows.append({
                    "Destination": c_item["dst"],
                    "Nom": dst_meta.get("name", "N/A"),
                    "Vols": c_item["flights"],
                })
            st.dataframe(
                pd.DataFrame(conn_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.info(
            f"La suppression de {selected_code} retirerait {s.get('degree_total', 0):,} "
            f"routes et {s.get('flights_total', 0):,} vols du reseau. "
            f"D'apres l'analyse de resilience, le reseau resterait largement "
            f"fonctionnel grace a sa haute redondance (84.1% de connectivite "
            f"apres 50 retraits cibles)."
        )

    with tab2:
        st.markdown("### Simulation de perturbation sequentielle")
        st.markdown(
            "Retrait progressif des aeroports les plus critiques (par betweenness) "
            "vs retrait aleatoire."
        )

        # Slider for max removal
        max_show = st.slider("Nombre d'aeroports retires", 5, 50, 50, key="disruption_slider")

        targeted = resilience["disruption"]["targeted"][:max_show + 1]
        random_ = resilience["disruption"]["random"][:max_show + 1]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Connectivite (%)", "Fragmentation (SCCs)"],
        )

        fig.add_trace(go.Scatter(
            x=[r["removed"] for r in targeted],
            y=[r["connectivity"] * 100 for r in targeted],
            mode="lines+markers", name="Ciblee (betweenness)",
            line=dict(color="#E91E63", width=2.5), marker=dict(size=4),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[r["removed"] for r in random_],
            y=[r["connectivity"] * 100 for r in random_],
            mode="lines+markers", name="Aleatoire",
            line=dict(color="#2196F3", width=2.5), marker=dict(size=4),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[r["removed"] for r in targeted],
            y=[r["num_scc"] for r in targeted],
            mode="lines+markers", name="Ciblee",
            line=dict(color="#E91E63", width=2.5), marker=dict(size=4),
            showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[r["removed"] for r in random_],
            y=[r["num_scc"] for r in random_],
            mode="lines+markers", name="Aleatoire",
            line=dict(color="#2196F3", width=2.5), marker=dict(size=4),
            showlegend=False,
        ), row=1, col=2)

        fig.update_xaxes(title_text="Aeroports retires")
        fig.update_yaxes(title_text="% du reseau", row=1, col=1)
        fig.update_yaxes(title_text="Nombre de SCC", row=1, col=2)
        fig.update_layout(height=450, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        drop_t = (targeted[0]["connectivity"] - targeted[-1]["connectivity"]) * 100
        drop_r = (random_[0]["connectivity"] - random_[-1]["connectivity"]) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Baisse ciblee ({max_show} retraits)", f"{drop_t:.2f} pp")
        c2.metric(f"Baisse aleatoire ({max_show} retraits)", f"{drop_r:.2f} pp")
        if drop_r > 0:
            c3.metric("Ratio cible/aleatoire", f"{drop_t / drop_r:.1f}x")

        st.success(
            "Le reseau est **remarquablement resilient**: meme l'attaque "
            "ciblee ne fragmente pas significativement le coeur du reseau, "
            "grace a la structure hub-and-spoke distribuee."
        )


# ===================================================================
# PAGE 5: MODEL PERFORMANCE
# ===================================================================
def page_model_performance():
    st.markdown("# 📊 Model Performance")
    st.markdown("*Comparaison detaillee de tous les modeles*")

    test = load_json("test_results.json")
    val = load_json("baseline_results.json")
    tgn_log = load_json("tgn_training_log.json")
    tgn_res = load_json("tgn_results.json")

    if not test or not val:
        st.error("Resultats non disponibles.")
        return

    val["TGN"] = {
        "mrr": tgn_log[-1]["val_mrr"],
        "hits@10": tgn_log[-1]["val_hits10"],
    }

    tab1, tab2, tab3, tab4 = st.tabs([
        "Resultats test",
        "Validation vs Test",
        "TGN Entrainement",
        "Tableau comparatif",
    ])

    with tab1:
        st.markdown("### Resultats sur le set de test")
        st.markdown("Donnees jamais vues pendant l'entrainement ou la validation.")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Mean Reciprocal Rank (MRR)", "Hits@10"],
        )
        for i, metric in enumerate(["mrr", "hits@10"]):
            vals = [test[m][metric] for m in MODELS]
            fig.add_trace(go.Bar(
                x=MODELS, y=vals,
                marker_color=[COLORS[m] for m in MODELS],
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
                showlegend=False,
            ), row=1, col=i + 1)
        fig.update_yaxes(range=[0, 1.05], row=1, col=1)
        fig.update_yaxes(range=[0, 1.1], row=1, col=2)
        fig.update_layout(height=450, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.info(
                "**HistFreq** domine le MRR (0.738): la frequence historique des "
                "routes est le signal le plus fort dans le reseau aerien."
            )
        with col2:
            st.info(
                "**TGN** a le meilleur Hits@10 (0.938): il identifie bien les "
                "destinations plausibles, meme si son ranking fin est moins precis."
            )

    with tab2:
        st.markdown("### Generalisation temporelle: Validation → Test")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["MRR", "Hits@10"],
        )
        for i, metric in enumerate(["mrr", "hits@10"]):
            v_vals = [val[m][metric] for m in MODELS]
            t_vals = [test[m][metric] for m in MODELS]
            fig.add_trace(go.Bar(
                x=MODELS, y=v_vals, name="Validation",
                marker_color="#64B5F6", showlegend=(i == 0),
            ), row=1, col=i + 1)
            fig.add_trace(go.Bar(
                x=MODELS, y=t_vals, name="Test",
                marker_color="#E91E63", showlegend=(i == 0),
            ), row=1, col=i + 1)
        fig.update_yaxes(range=[0, 1.1])
        fig.update_layout(barmode="group", height=450, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Gap metrics
        st.markdown("#### Gap MRR: Validation → Test")
        cols = st.columns(5)
        for col, m in zip(cols, MODELS):
            delta = test[m]["mrr"] - val[m]["mrr"]
            col.metric(m, f"{test[m]['mrr']:.3f}", f"{delta:+.3f}")

        st.warning(
            "**TGN** a le plus grand gap (-0.079): entraine sur 3 mois de 2021, "
            "il generalise moins bien a 2022. **Recency** s'ameliore (+0.022) "
            "grace aux timestamps plus recents."
        )

    with tab3:
        st.markdown("### Courbes d'entrainement TGN")

        if tgn_log:
            epochs = [e["epoch"] for e in tgn_log]
            losses = [e["train_loss"] for e in tgn_log]
            mrrs = [e["val_mrr"] for e in tgn_log]
            hits = [e["val_hits10"] for e in tgn_log]

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Loss BPR", "Metriques de validation"],
            )
            fig.add_trace(go.Scatter(
                x=epochs, y=losses, mode="lines+markers",
                name="Train Loss", line=dict(color="#E91E63", width=3),
                marker=dict(size=12),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=epochs, y=mrrs, mode="lines+markers",
                name="Val MRR", line=dict(color="#2196F3", width=3),
                marker=dict(size=12),
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=epochs, y=hits, mode="lines+markers",
                name="Val Hits@10", line=dict(color="#4CAF50", width=3),
                marker=dict(size=12),
            ), row=1, col=2)
            fig.update_xaxes(title_text="Epoch", dtick=1)
            fig.update_layout(height=400, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # TGN config
        if tgn_res:
            st.markdown("#### Configuration du modele")
            c1, c2, c3, c4 = st.columns(4)
            config = tgn_res.get("config", {})
            c1.metric("Parametres", f"{tgn_res.get('total_params', 0):,}")
            c2.metric("Memory dim", config.get("memory_dim", "N/A"))
            c3.metric("Batch size", config.get("batch_size", "N/A"))
            c4.metric("Epochs", config.get("epochs", "N/A"))

            st.markdown("#### Temps d'execution")
            c1, c2, c3 = st.columns(3)
            c1.metric("Entrainement (2 epochs)", "~27 min")
            c2.metric("Evaluation validation", "~2 min")
            c3.metric("Evaluation test (warmup)", "~56 min")

    with tab4:
        st.markdown("### Tableau comparatif complet")

        rows = []
        for m in MODELS:
            rows.append({
                "Modele": m,
                "Type": MODEL_TYPES[m],
                "Val MRR": val[m]["mrr"],
                "Test MRR": test[m]["mrr"],
                "Delta MRR": test[m]["mrr"] - val[m]["mrr"],
                "Val Hits@10": val[m]["hits@10"],
                "Test Hits@10": test[m]["hits@10"],
            })
        df = pd.DataFrame(rows)

        st.dataframe(
            df.style.format({
                "Val MRR": "{:.4f}",
                "Test MRR": "{:.4f}",
                "Delta MRR": "{:+.4f}",
                "Val Hits@10": "{:.4f}",
                "Test Hits@10": "{:.4f}",
            }).highlight_max(
                subset=["Test MRR", "Test Hits@10"], color="#C8E6C9"
            ).highlight_min(
                subset=["Delta MRR"], color="#FFCDD2"
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Analyse")
        st.markdown("""
| Observation | Explication |
|-------------|-------------|
| HistFreq #1 en MRR | Le reseau aerien est extremement regulier — les routes stables dominent |
| TGN #1 en Hits@10 | TGN discrimine bien les destinations plausibles vs impossibles |
| TGN gap val→test le plus grand | Entraine sur 3 mois de 2021, evalue sur 2022 (>1 an d'ecart) |
| Recency s'ameliore sur test | Beneficie des timestamps plus recents de train+val |
| EdgeBank faible en MRR | Ponderation binaire (existe/n'existe pas) perd l'information de frequence |
        """)


# ===================================================================
# Router
# ===================================================================
if page == "🌍 Overview":
    page_overview()
elif page == "🔎 Airport Explorer":
    page_airport_explorer()
elif page == "🔮 Future Link Predictions":
    page_predictions()
elif page == "🛡️ Network Resilience Simulator":
    page_resilience_simulator()
elif page == "📊 Model Performance":
    page_model_performance()
