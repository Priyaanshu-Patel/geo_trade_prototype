"""
Correlation graph: builds asset network and propagates event cascades.
Nodes = assets. Edges = rolling correlation (|r| > threshold).
Event impact on directly-affected assets flows to neighbors via edge weights.
"""
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

CORR_THRESHOLD = 0.25   # Min |correlation| to keep an edge
DECAY = 0.65            # Cascade decay per hop


def build_graph(correlation_matrix: pd.DataFrame, asset_meta: dict) -> nx.Graph:
    """Build weighted undirected graph from correlation matrix."""
    G = nx.Graph()

    tickers = [t for t in correlation_matrix.columns if not correlation_matrix[t].isna().all()]
    for t in tickers:
        meta = asset_meta.get(t, {})
        G.add_node(t, sector=meta.get("sector", "other"), region=meta.get("region", "global"), name=meta.get("name", t))

    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i >= j:
                continue
            corr = correlation_matrix.loc[t1, t2]
            if pd.isna(corr) or abs(corr) < CORR_THRESHOLD:
                continue
            G.add_edge(t1, t2, weight=float(corr), abs_weight=abs(float(corr)))

    return G


def propagate_cascade(
    G: nx.Graph,
    direct_impacts: dict,  # ticker -> predicted return (fraction, e.g. 0.02)
) -> dict:
    """
    Spread direct event impacts through the correlation graph.
    Returns full impact dict for all nodes (direct + propagated).
    """
    impacts = {n: 0.0 for n in G.nodes()}

    # Apply direct impacts
    for ticker, impact in direct_impacts.items():
        if ticker in impacts:
            impacts[ticker] = impact

    # One-hop propagation to neighbors
    propagated = dict(impacts)
    for ticker, direct_impact in impacts.items():
        if abs(direct_impact) < 1e-6:
            continue
        for neighbor in G.neighbors(ticker):
            edge_weight = G[ticker][neighbor]["weight"]
            # Sign-aware: positive corr → same direction; negative corr → opposite
            propagated[neighbor] += DECAY * edge_weight * direct_impact

    # Clip to reasonable range (max ±10% single-event impact)
    return {t: max(-0.10, min(0.10, v)) for t, v in propagated.items()}


def graph_summary(G: nx.Graph, impacts: Optional[dict] = None) -> str:
    """Plain-text graph description for LLM context."""
    lines = [f"Assets: {list(G.nodes())}"]
    lines.append(f"Edges (|corr| > {CORR_THRESHOLD}):")
    edges = sorted(G.edges(data=True), key=lambda x: -abs(x[2]["weight"]))
    for u, v, d in edges[:15]:
        lines.append(f"  {u} <-> {v}: corr={d['weight']:.2f}")
    if impacts:
        lines.append("\nCascade impacts:")
        for t, imp in sorted(impacts.items(), key=lambda x: -abs(x[1])):
            if abs(imp) > 0.001:
                lines.append(f"  {t}: {imp*100:+.2f}%")
    return "\n".join(lines)


def graph_to_plotly(G: nx.Graph, impacts: Optional[dict] = None, title: str = "Asset Correlation Network") -> go.Figure:
    """
    Render the NetworkX graph as an interactive Plotly figure.
    Node color = sector. Node size = |cascade impact| if provided.
    Edge color = correlation sign (green positive, red negative).
    """
    if len(G.nodes()) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=42, k=2.5)

    SECTOR_COLORS = {
        "tech":     "#4A90E2",
        "energy":   "#F5A623",
        "defense":  "#7B68EE",
        "gold":     "#FFD700",
        "oil":      "#CD853F",
        "currency": "#2ECC71",
        "other":    "#999999",
    }

    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = "rgba(0,180,0,0.4)" if data["weight"] > 0 else "rgba(220,0,0,0.4)"
        width = 1 + 3 * data["abs_weight"]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="none",
                showlegend=False,
            )
        )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        sector = G.nodes[node].get("sector", "other")
        node_color.append(SECTOR_COLORS.get(sector, "#999"))

        impact = impacts.get(node, 0.0) if impacts else 0.0
        node_size.append(20 + abs(impact) * 400)

        name = G.nodes[node].get("name", node)
        impact_str = f"<br>Impact: {impact*100:+.2f}%" if impacts else ""
        node_text.append(f"<b>{node}</b><br>{name}<br>Sector: {sector}{impact_str}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(color=node_color, size=node_size, line=dict(width=1.5, color="#333")),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
