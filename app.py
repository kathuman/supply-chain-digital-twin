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
        G.add_edge(row['source'], row['target'])
        
    return G

# --- 3. LOGIC & VISUALIZATION ---
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
st.title("🌐 Supply Chain Digital Twin: 4-Tier Network Planner")

st.sidebar.header("Data Input")
upload_file = st.sidebar.file_uploader("Upload Supply Chain Excel", type=["xlsx"])
demand_slider = st.sidebar.slider("Monthly Demand Units", 100, 2000, 500)

# Default Sample Data
if upload_file:
    nodes_df = pd.read_excel(upload_file, sheet_name='Nodes')
    edges_df = pd.read_excel(upload_file, sheet_name='Edges')
else:
    st.sidebar.info("Using Sample Data. Upload an Excel file to customize.")
    nodes_df = pd.DataFrame({
        'id': ['Raw_China', 'Raw_Germany', 'Assy_Mexico', 'WH_Dallas', 'Retail_NY'],
        'type': ['manufacturer', 'manufacturer', 'manufacturer', 'warehouse', 'demand'],
        'tier': [1, 1, 2, 2.5, 3],
        'lat': [31.2, 52.5, 19.4, 32.7, 40.7], 'lon': [121.4, 13.4, -99.1, -96.7, -74.0],
        'max_capacity': [1000, 1000, 600, None, None],
        'capacity': [None, None, None, 7000, None],
        'materials': ['M1', 'M2', 'Prod_A', '', 'Prod_A'],
        'prod_time': [10, 5, 12, 0, 0], 'min_batch': [100, 50, 20, 1, 1], 'cost': [5, 5, 20, 2, 0]
    })
    edges_df = pd.DataFrame({'source': ['Raw_China', 'Raw_Germany', 'Assy_Mexico', 'WH_Dallas'], 
                             'target': ['Assy_Mexico', 'Assy_Mexico', 'WH_Dallas', 'Retail_NY']})

network = build_network_from_tables(nodes_df, edges_df)
run_utilization_analysis(network, demand_slider)

tab1, tab2, tab3 = st.tabs(["📊 Logical Heat Map", "🌍 Geospatial Map", "📋 Raw Data"])

with tab1:
    st.subheader("Network Capacity Utilization")
    pos = nx.multipartite_layout(network, subset_key="layer")
    node_x, node_y, node_colors, hover_texts = [], [], [], []
    for node, data in network.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        util = getattr(data['obj'], 'peak_utilization', 0)
        node_colors.append(util)
        hover_texts.append(f"Node: {node}<br>Util: {util:.1f}%")

    fig_heat = go.Figure(data=[go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=hover_texts,
        marker=dict(showscale=True, colorscale='RdYlGn_r', color=node_colors, size=25, colorbar=dict(title="% Util"))
    )], layout=go.Layout(plot_bgcolor='white', xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)))
    st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    st.subheader("Global Facility Footprint")
    map_data = []
    for node, data in network.nodes(data=True):
        obj = data['obj']
        map_data.append({'Node': node, 'Lat': obj.lat, 'Lon': obj.lon, 'Util': getattr(obj, 'peak_utilization', 0)})
    df_map = pd.DataFrame(map_data)
    fig_map = px.scatter_geo(df_map, lat='Lat', lon='Lon', color='Util', hover_name='Node', color_continuous_scale='Reds')
    
    for s, t in network.edges():
        s_obj, t_obj = network.nodes[s]['obj'], network.nodes[t]['obj']
        fig_map.add_trace(go.Scattergeo(lat=[s_obj.lat, t_obj.lat], lon=[s_obj.lon, t_obj.lon], mode='lines', line=dict(width=1, color='gray'), opacity=0.3))
    
    fig_map.update_geos(projection_type="natural earth", showcountries=True)
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.dataframe(nodes_df)
