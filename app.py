"""
Delhivery Network Intelligence — live delay-risk dashboard
==========================================================
Optional Streamlit deliverable for "Optimizing Delivery ETAs with Graph-Based
Network Intelligence". Driven by the SAME pipeline as delhivery_graph_eta.ipynb:
segments -> OD legs -> corridors -> directed weighted graph -> centrality, SLA-breach
risk scores, and a graph-enhanced LightGBM ETA model.

Run:  streamlit run app.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Delhivery Network Intelligence",
                   page_icon="🚚", layout="wide")
DATA = "delivery_data.csv"
RND = 42


# --------------------------------------------------------------------------- #
#  Pipeline (cached) — identical logic to the notebook
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading & processing delivery data…")
def load_pipeline():
    df = pd.read_csv(DATA)
    for c in ["od_start_time", "od_end_time"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    cum = ["actual_time", "osrm_time", "osrm_distance",
           "actual_distance_to_destination", "factor", "start_scan_to_end_scan"]
    seg = ["segment_actual_time", "segment_osrm_time", "segment_osrm_distance"]
    keys = ["trip_uuid", "source_center", "destination_center", "od_start_time"]
    agg = {**{c: "max" for c in cum}, **{c: "sum" for c in seg},
           "route_type": "first", "data": "first",
           "source_name": "first", "destination_name": "first"}
    od = df.groupby(keys, as_index=False).agg(agg)
    od["n_segments"] = df.groupby(keys).size().values

    od["hour"] = od.od_start_time.dt.hour
    od["weekday"] = od.od_start_time.dt.dayofweek
    od["tod"] = pd.cut(od.hour, [-1, 5, 11, 16, 21, 24],
                       labels=["night", "morning", "afternoon", "evening", "late_night"])
    od["delay_ratio"] = od.actual_time / od.osrm_time.replace(0, np.nan)
    od = od.dropna(subset=["actual_time", "osrm_time", "osrm_distance"])
    od = od[od.osrm_time > 0].reset_index(drop=True)

    corr = (od.groupby(["source_center", "destination_center"])
              .agg(trips=("actual_time", "size"),
                   med_ratio=("delay_ratio", "median"),
                   med_actual=("actual_time", "median"),
                   med_osrm=("osrm_time", "median"),
                   med_dist=("osrm_distance", "median"))
              .reset_index())
    corr["chronic"] = corr.med_ratio > 1.20
    corr["breach"] = corr.trips * (corr.med_ratio - 1).clip(lower=0)

    name_map = (pd.concat([
        od[["source_center", "source_name"]].rename(
            columns={"source_center": "node", "source_name": "name"}),
        od[["destination_center", "destination_name"]].rename(
            columns={"destination_center": "node", "destination_name": "name"})])
        .dropna().drop_duplicates("node").set_index("node")["name"])
    corr["src_name"] = corr.source_center.map(name_map).fillna(corr.source_center)
    corr["dst_name"] = corr.destination_center.map(name_map).fillna(corr.destination_center)
    return od, corr, name_map


@st.cache_resource(show_spinner="Building graph & training ETA model…")
def build_graph_model(_od, _corr):
    od, corr = _od, _corr
    G = nx.DiGraph()
    for r in corr.itertuples(index=False):
        G.add_edge(r.source_center, r.destination_center,
                   weight=r.med_ratio, trips=r.trips)

    btw = nx.betweenness_centrality(G, weight="weight",
                                    k=min(400, G.number_of_nodes()), seed=RND)
    ind, outd = dict(G.in_degree()), dict(G.out_degree())
    clus = nx.clustering(G.to_undirected())
    nodes = list(G.nodes())
    ndf = pd.DataFrame({"node": nodes,
                        "btw": [btw.get(n, 0) for n in nodes],
                        "in_deg": [ind.get(n, 0) for n in nodes],
                        "out_deg": [outd.get(n, 0) for n in nodes],
                        "clus": [clus.get(n, 0) for n in nodes]})
    hub_breach = (corr.groupby("source_center").breach.sum()
                  .add(corr.groupby("destination_center").breach.sum(), fill_value=0))
    ndf["breach"] = ndf.node.map(hub_breach).fillna(0)
    # 0-100 delay-risk score: realised breach blended with structural centrality
    b = ndf.breach / ndf.breach.max() if ndf.breach.max() else ndf.breach
    c = ndf.btw / ndf.btw.max() if ndf.btw.max() else ndf.btw
    ndf["risk"] = (100 * (0.7 * b + 0.3 * c)).round(1)

    # graph-enhanced ETA model (base + centrality features — fast, no embeddings at predict time)
    gmap = ndf.set_index("node")
    od = od.copy()
    od["rt"] = (od.route_type == "FTL").astype(int)
    for col in ["btw", "in_deg", "out_deg", "clus"]:
        od["src_" + col] = od.source_center.map(gmap[col]).fillna(0)
        od["dst_" + col] = od.destination_center.map(gmap[col]).fillna(0)
    base = ["osrm_time", "osrm_distance", "actual_distance_to_destination",
            "rt", "hour", "weekday", "n_segments"]
    gfeat = base + ["src_btw", "src_in_deg", "src_out_deg", "src_clus",
                    "dst_btw", "dst_in_deg", "dst_out_deg", "dst_clus"]
    tr = (od.data == "training").values
    te = (od.data == "test").values
    y = od.actual_time.values

    def fit(feats):
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=64,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RND, n_jobs=-1, verbose=-1)
        m.fit(od.loc[tr, feats], y[tr])
        p = m.predict(od.loc[te, feats])
        w15 = float(np.mean(np.abs(p - y[te]) / np.clip(y[te], 1e-6, None) <= 0.15) * 100)
        return m, float(mean_absolute_error(y[te], p)), w15

    m_base, mae_b, w_b = fit(base)
    m_graph, mae_g, w_g = fit(gfeat)
    metrics = dict(mae_base=mae_b, w_base=w_b, mae_graph=mae_g, w_graph=w_g,
                   osrm_w=float(np.mean(np.abs(od.loc[te, "osrm_time"] - y[te])
                                        / np.clip(y[te], 1e-6, None) <= 0.15) * 100))

    # one stable spring layout over the top-breach subgraph (for the map)
    top = corr.sort_values("breach", ascending=False).head(500)
    H = nx.from_pandas_edgelist(top, "source_center", "destination_center",
                                create_using=nx.DiGraph())
    pos = nx.spring_layout(H, k=0.4, seed=RND, iterations=50)
    return ndf, m_graph, gfeat, metrics, pos


# --------------------------------------------------------------------------- #
#  Load
# --------------------------------------------------------------------------- #
od, corr, name_map = load_pipeline()
ndf, model, gfeat, metrics, pos = build_graph_model(od, corr)

st.title("🚚 Delhivery Network Intelligence")
st.caption("Live delay-risk scores across the logistics graph — facilities as nodes, "
           "corridors as edges. Same pipeline as `delhivery_graph_eta.ipynb`.")

# --------------------------------------------------------------------------- #
#  Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.header("Filters")
rtypes = st.sidebar.multiselect("Route type", sorted(od.route_type.unique()),
                                default=sorted(od.route_type.unique()))
min_trips = st.sidebar.slider("Min corridor volume (trips)", 1, 50, 3)
only_chronic = st.sidebar.checkbox("Only chronically-delayed corridors (>20%)", False)
top_n = st.sidebar.slider("Corridors on network map", 50, 500, 250, step=50)

cf = corr[corr.trips >= min_trips].copy()
if only_chronic:
    cf = cf[cf.chronic]
if rtypes:
    valid = set(od[od.route_type.isin(rtypes)]
                .groupby(["source_center", "destination_center"]).groups.keys())
    cf = cf[cf.set_index(["source_center", "destination_center"]).index.isin(valid)]

# --------------------------------------------------------------------------- #
#  KPI row
# --------------------------------------------------------------------------- #
k = st.columns(5)
k[0].metric("Facilities (nodes)", f"{ndf.node.nunique():,}")
k[1].metric("Corridors (edges)", f"{len(cf):,}")
k[2].metric("Median delay ratio", f"{od.delay_ratio.median():.2f}×")
k[3].metric("Chronic corridors", f"{100*corr.chronic.mean():.0f}%")
k[4].metric("Graph ETA within-15%", f"{metrics['w_graph']:.0f}%",
            delta=f"{metrics['w_graph']-metrics['osrm_w']:.0f}pp vs OSRM")

tab_map, tab_hub, tab_corr, tab_eta, tab_model = st.tabs(
    ["🗺️ Network Map", "🏭 Hub Risk", "🛣️ Corridor Risk", "🔮 ETA Predictor", "📊 Model"])

# --------------------------------------------------------------------------- #
#  Network map
# --------------------------------------------------------------------------- #
with tab_map:
    st.subheader("Corridor network — node size/colour = delay-risk score")
    show = cf.sort_values("breach", ascending=False).head(top_n)
    nodes_in = set(show.source_center) | set(show.destination_center)
    nodes_in = [n for n in nodes_in if n in pos]
    risk = ndf.set_index("node")["risk"]

    ex, ey = [], []
    for r in show.itertuples(index=False):
        if r.source_center in pos and r.destination_center in pos:
            x0, y0 = pos[r.source_center]; x1, y1 = pos[r.destination_center]
            ex += [x0, x1, None]; ey += [y0, y1, None]
    edge_tr = go.Scatter(x=ex, y=ey, mode="lines",
                         line=dict(width=0.5, color="#cccccc"),
                         hoverinfo="none")
    nx_, ny_ = zip(*[pos[n] for n in nodes_in]) if nodes_in else ([], [])
    node_tr = go.Scatter(
        x=nx_, y=ny_, mode="markers",
        marker=dict(size=[6 + risk.get(n, 0) / 4 for n in nodes_in],
                    color=[risk.get(n, 0) for n in nodes_in],
                    colorscale="YlOrRd", showscale=True,
                    colorbar=dict(title="Risk"), line=dict(width=0.5, color="#444")),
        text=[f"{name_map.get(n, n)}<br>risk={risk.get(n,0):.0f}" for n in nodes_in],
        hoverinfo="text")
    fig = go.Figure([edge_tr, node_tr])
    fig.update_layout(showlegend=False, height=620, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white")
    st.plotly_chart(fig, width='stretch')
    st.caption("Hover a node for its facility name and delay-risk score. "
               "Layout reflects graph structure (not geography).")

# --------------------------------------------------------------------------- #
#  Hub risk
# --------------------------------------------------------------------------- #
with tab_hub:
    st.subheader("Facilities ranked by delay-risk score")
    hub = ndf.copy()
    hub["name"] = hub.node.map(name_map).fillna(hub.node)
    hub = hub.sort_values("risk", ascending=False)
    n_top = st.slider("Show top N hubs", 5, 50, 15, key="hubn")
    top = hub.head(n_top)
    fig = px.bar(top[::-1], x="risk", y="name", orientation="h",
                 color="risk", color_continuous_scale="YlOrRd",
                 labels={"risk": "delay-risk score", "name": ""}, height=28 * n_top + 80)
    st.plotly_chart(fig, width='stretch')
    st.dataframe(top[["name", "node", "risk", "breach", "btw", "in_deg", "out_deg"]]
                 .rename(columns={"btw": "betweenness", "in_deg": "in", "out_deg": "out"}),
                 width='stretch', hide_index=True)

    st.markdown("**Drill into a hub's corridors**")
    pick = st.selectbox("Facility", top.node.tolist(),
                        format_func=lambda n: f"{name_map.get(n, n)}")
    out_c = corr[corr.source_center == pick].assign(direction="outbound")
    in_c = corr[corr.destination_center == pick].assign(direction="inbound")
    drill = pd.concat([out_c, in_c]).sort_values("breach", ascending=False)
    st.dataframe(drill[["direction", "src_name", "dst_name", "trips",
                        "med_ratio", "med_actual", "med_osrm", "chronic"]].head(20),
                 width='stretch', hide_index=True)

# --------------------------------------------------------------------------- #
#  Corridor risk
# --------------------------------------------------------------------------- #
with tab_corr:
    st.subheader("Corridors by SLA-breach contribution")
    view = cf.sort_values("breach", ascending=False)
    st.plotly_chart(
        px.scatter(view.head(800), x="trips", y="med_ratio", size="breach",
                   color="med_ratio", color_continuous_scale="YlOrRd",
                   hover_data=["src_name", "dst_name"],
                   labels={"trips": "volume", "med_ratio": "delay ratio (actual/OSRM)"},
                   height=480).add_hline(y=1.20, line_dash="dash", line_color="grey"),
        width='stretch')
    st.dataframe(view[["src_name", "dst_name", "trips", "med_ratio",
                       "med_actual", "med_osrm", "med_dist", "breach", "chronic"]].head(50),
                 width='stretch', hide_index=True)

# --------------------------------------------------------------------------- #
#  ETA predictor
# --------------------------------------------------------------------------- #
with tab_eta:
    st.subheader("Graph-enhanced ETA predictor")
    st.caption("Predicts actual leg time from route inputs + the source/destination "
               "facilities' graph position. Compare against the naive OSRM estimate.")
    gmap = ndf.set_index("node")
    opts = ndf.sort_values("risk", ascending=False).node.tolist()
    c1, c2 = st.columns(2)
    src = c1.selectbox("Source facility", opts,
                       format_func=lambda n: name_map.get(n, n))
    dst = c2.selectbox("Destination facility", opts, index=min(1, len(opts) - 1),
                       format_func=lambda n: name_map.get(n, n))
    c3, c4, c5 = st.columns(3)
    osrm_t = c3.number_input("OSRM time (min)", 1.0, 2000.0, 60.0, 5.0)
    osrm_d = c4.number_input("OSRM distance (km)", 1.0, 3000.0, 50.0, 5.0)
    rtype = c5.selectbox("Route type", ["FTL", "Carting"])
    c6, c7, c8 = st.columns(3)
    act_d = c6.number_input("Actual distance to dest (km)", 1.0, 3000.0, 55.0, 5.0)
    hour = c7.slider("Departure hour", 0, 23, 10)
    nseg = c8.slider("# segments", 1, 30, 3)

    row = {"osrm_time": osrm_t, "osrm_distance": osrm_d,
           "actual_distance_to_destination": act_d, "rt": int(rtype == "FTL"),
           "hour": hour, "weekday": 2, "n_segments": nseg}
    for col in ["btw", "in_deg", "out_deg", "clus"]:
        row["src_" + col] = float(gmap.loc[src, col]) if src in gmap.index else 0.0
        row["dst_" + col] = float(gmap.loc[dst, col]) if dst in gmap.index else 0.0
    pred = float(model.predict(pd.DataFrame([row])[gfeat])[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("OSRM estimate", f"{osrm_t:.0f} min")
    m2.metric("Graph-enhanced ETA", f"{pred:.0f} min", delta=f"{pred-osrm_t:+.0f} min")
    m3.metric("Predicted delay ratio", f"{pred/osrm_t:.2f}×")
    if pred > osrm_t * 1.2:
        st.warning("⚠️ High delay risk — model expects >20% over the OSRM estimate.")
    else:
        st.success("✅ Within ~20% of the OSRM estimate.")

# --------------------------------------------------------------------------- #
#  Model comparison
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("Baseline vs graph-enhanced (held-out test split)")
    comp = pd.DataFrame({
        "model": ["OSRM (naive)", "Baseline ML", "Graph-enhanced"],
        "within15": [metrics["osrm_w"], metrics["w_base"], metrics["w_graph"]],
        "MAE": [np.nan, metrics["mae_base"], metrics["mae_graph"]]})
    cc = st.columns(2)
    cc[0].plotly_chart(px.bar(comp, x="model", y="within15", color="model",
                              title="% trips within 15% of actual (higher better)",
                              labels={"within15": "%"}), width='stretch')
    cc[1].plotly_chart(px.bar(comp.dropna(), x="model", y="MAE", color="model",
                              title="MAE (lower better)"), width='stretch')
    st.info(f"Graph features cut MAE {metrics['mae_base']:.1f} → {metrics['mae_graph']:.1f} "
            f"({100*(metrics['mae_base']-metrics['mae_graph'])/metrics['mae_base']:+.0f}%) "
            f"and lifted within-15% accuracy "
            f"{metrics['w_base']:.0f}% → {metrics['w_graph']:.0f}% "
            f"(vs ~{metrics['osrm_w']:.0f}% for naive OSRM). "
            "Full node2vec model is in the notebook.")
