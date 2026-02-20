import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import math
from dataclasses import dataclass
from typing import List

# --- 1. DATA STRUCTURES ---
@dataclass
class ManufacturingSite:
    id: str
    tier: float
    lat: float
    lon: float
    materials_produced: List[str]
    production_lead_time: float
    min_batch: int
    max_capacity: float
    cost_per_unit: float
    peak_utilization: float = 0.0

@dataclass
class Warehouse:
    id: str
    tier: float
    lat: float
    lon: float
    capacity: float
    cost_per_unit: float
    peak_utilization: float = 0.0

@dataclass
class DemandPoint:
    id: str
    product_id: str
    lat: float
    lon: float

# --- 2. NETWORK BUILDER ---
def build_network_from_tables(nodes_df, edges_df):
    G = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        node_id = row['id']
        n_type = row['type']
        
        if n_type == 'manufacturer':
            obj = ManufacturingSite(
                id=node_id, tier=row['tier'],
                lat=row.get('lat', 0), lon=row.get('lon', 0),
                materials_produced=[m.strip() for m in str(row.get('materials', '')).split(',')],
                production_lead_time=row.get('prod_time', 0),
                min_batch=int(row.get('min_batch', 1)),
                max_capacity=row.get('max_capacity', 999999),
                cost_per_unit=row.get('cost', 0)
            )
        elif n_type == 'warehouse':
            obj = Warehouse(
                id=node_id, tier=row['tier'],
                lat=row.get('lat', 0), lon=row.get('lon', 0),
                capacity=row.get('capacity', 999999),
                cost_per_unit=row.get('cost', 0)
            )
        elif n_type == 'demand':
            obj = DemandPoint(id=node_id, product_id=row.get('materials', ''), 
                              lat=row.get('lat', 0), lon=row.get('lon', 0))
        
        G.add_node(node_id, type=n_type, obj=obj, layer=row['tier'])
    
    for _, row in edges_df.iterrows():
        # Store transport info as edge attributes
        G.add_edge(row['source'], row['target'], 
                   lead_time=row.get('transit_time', 0), 
                   cost=row.get('transit_cost', 0))
        
    return G

# --- 3. UTILIZATION LOGIC ---
def run_utilization_analysis(G, monthly_demand):
    target_moh = 12
    target_stock = monthly_demand * target_moh
    for node, data in G.nodes(data=True):
        obj = data['obj']
        if data['type'] == 'warehouse':
            obj.peak_utilization = (target_stock / obj.capacity) * 100 if obj.capacity > 0 else 0
        elif data['type'] == 'manufacturer':
            obj.peak_utilization = (monthly_demand / obj.max_capacity) * 100 if obj.max_capacity > 0 else 0

# --- 4. STREAMLIT APP UI ---
st.set_page_config(page_title="Supply Chain Digital Twin", layout="wide")
st.title("🌐 Supply Chain Twin: Interactive Hover & Heatmap")

st.sidebar.header("Data Management")
upload_file = st.sidebar.file_uploader("Upload Supply Chain Excel", type=["xlsx"])
demand_slider = st.sidebar.slider("Monthly Demand Units", 100, 2000, 500)

if upload_file:
    nodes_df = pd.read_excel(upload_file, sheet_name='Nodes')
    edges_df = pd.read_excel(upload_file, sheet_name='Edges')
else:
    st.sidebar.info("Using Sample Data.")
    nodes_df = pd.DataFrame({
        'id': ['Factory_A', 'WH_1', 'Retail_X'],
        'type': ['manufacturer', 'warehouse', 'demand'],
        'tier': [1, 2, 3], 'lat': [35.6, 40.7, 34.0], 'lon': [139.6, -74.0, -118.2],
        'max_capacity': [1000, None, None], 'capacity': [None, 7000, None],
        'materials': ['P1', '', 'P1'], 'prod_time': [10, 0, 0], 'min_batch': [100, 1, 1], 'cost': [20, 2, 0]
    })
    edges_df = pd.DataFrame({
        'source': ['Factory_A', 'WH_1'], 'target': ['WH_1', 'Retail_X'],
        'transit_time': [14, 3], 'transit_cost': [5.5, 2.0]
    })

network = build_network_from_tables(nodes_df, edges_df)
run_utilization_analysis(network, demand_slider)

tab1, tab2 = st.tabs(["📊 Logical Network", "🌍 Global Map"])

# --- TAB 1: LOGICAL HEAT MAP WITH EDGE HOVER ---
with tab1:
    pos = nx.multipartite_layout(network, subset_key="layer")
    
    # Draw Edge Lines
    edge_x, edge_y = [], []
    for s, t in network.edges():
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_line_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#bbb'), mode='lines', hoverinfo='none', showlegend=False)

    # Draw Edge Midpoints for Hover
    mid_x, mid_y, mid_text = [], [], []
    for s, t, d in network.edges(data=True):
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_text.append(f"<b>Route: {s} → {t}</b><br>Lead Time: {d['lead_time']} days<br>Transport Cost: ${d['cost']}")

    edge_hover_trace = go.Scatter(x=mid_x, y=mid_y, mode='markers', marker=dict(size=10, color='rgba(0,0,0,0)'), text=mid_text, hoverinfo='text', showlegend=False)

    # Nodes
    node_x, node_y, node_util, node_text = [], [], [], []
    for n, d in network.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        util = getattr(d['obj'], 'peak_utilization', 0)
        node_util.append(util)
        node_text.append(f"Node: {n}<br>Util: {util:.1f}%")

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(showscale=True, colorscale='RdYlGn_r', color=node_util, size=22, colorbar=dict(title="% Util")), text=node_text, hoverinfo='text')

    fig_logical = go.Figure(data=[edge_line_trace, edge_hover_trace, node_trace], layout=go.Layout(plot_bgcolor='white', xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    st.plotly_chart(fig_logical, use_container_width=True)

# --- TAB 2: GEOSPATIAL MAP WITH EDGE HOVER ---
with tab2:
    fig_map = go.Figure()

    # Draw Edge Lines & Midpoints
    for s, t, d in network.edges(data=True):
        s_obj, t_obj = network.nodes[s]['obj'], network.nodes[t]['obj']
        
        # Line
        fig_map.add_trace(go.Scattergeo(lat=[s_obj.lat, t_obj.lat], lon=[s_obj.lon, t_obj.lon], mode='lines', line=dict(width=1, color='gray'), opacity=0.4, showlegend=False, hoverinfo='none'))
        
        # Midpoint Hover
        fig_map.add_trace(go.Scattergeo(lat=[(s_obj.lat + t_obj.lat)/2], lon=[(s_obj.lon + t_obj.lon)/2], mode='markers', marker=dict(size=8, color='rgba(0,0,0,0)'), text=f"<b>{s} → {t}</b><br>Time: {d['lead_time']}d<br>Cost: ${d['cost']}", hoverinfo='text', showlegend=False))

    # Nodes
    map_lat, map_lon, map_util, map_hover = [], [], [], []
    for n, d in network.nodes(data=True):
        map_lat.append(d['obj'].lat)
        map_lon.append(d['obj'].lon)
        u = getattr(d['obj'], 'peak_utilization', 0)
        map_util.append(u)
        map_hover.append(f"{n} ({u:.1f}% Util)")

    fig_map.add_trace(go.Scattergeo(lat=map_lat, lon=map_lon, mode='markers', marker=dict(size=12, color=map_util, colorscale='Reds', showscale=True), text=map_hover, hoverinfo='text', name="Facilities"))

    fig_map.update_geos(projection_type="natural earth", showcountries=True, landcolor="#f9f9f9")
    fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
