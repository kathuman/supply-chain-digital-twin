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
st.title("🌐 Supply Chain Twin: End-to-End Strategic Planner")

# Sidebar for controls
st.sidebar.header("Control Center")
upload_file = st.sidebar.file_uploader("Upload Supply Chain Excel", type=["xlsx"])
demand_slider = st.sidebar.slider("Global Monthly Demand", 100, 5000, 500)

# Load Data Logic
if upload_file:
    nodes_df = pd.read_excel(upload_file, sheet_name='Nodes')
    edges_df = pd.read_excel(upload_file, sheet_name='Edges')
else:
    # Embedded Sample Data
    nodes_df = pd.DataFrame({
        'id': ['Raw_China', 'Raw_Germany', 'Assy_Mexico', 'WH_Dallas', 'Retail_NY'],
        'type': ['manufacturer', 'manufacturer', 'manufacturer', 'warehouse', 'demand'],
        'tier': [1, 1, 2, 2.5, 3],
        'lat': [31.2, 52.5, 19.4, 32.7, 40.7], 'lon': [121.4, 13.4, -99.1, -96.7, -74.0],
        'max_capacity': [2000, 2000, 1000, None, None],
        'capacity': [None, None, None, 10000, None],
        'materials': ['M1', 'M2', 'Prod_A', '', 'Prod_A'],
        'prod_time': [10, 5, 12, 0, 0], 'min_batch': [100, 50, 20, 1, 1], 'cost': [5, 5, 20, 2, 0]
    })
    edges_df = pd.DataFrame({
        'source': ['Raw_China', 'Raw_Germany', 'Assy_Mexico', 'WH_Dallas'], 
        'target': ['Assy_Mexico', 'Assy_Mexico', 'WH_Dallas', 'Retail_NY'],
        'transit_time': [25, 18, 5, 3], 'transit_cost': [4.5, 3.0, 1.2, 0.8]
    })

network = build_network_from_tables(nodes_df, edges_df)
run_utilization_analysis(network, demand_slider)

# Tab Navigation
tab_docs, tab_logic, tab_geo, tab_data = st.tabs([
    "📖 Documentation", 
    "📊 Logical Network", 
    "🌍 Global Map", 
    "📋 Raw Data Explorer"
])

# --- TAB: DOCUMENTATION ---
with tab_docs:
    st.header("App Overview & Strategic Logic")
    st.markdown("""
    ### Purpose
    This Digital Twin simulates a **4-tier supply chain network** to evaluate the feasibility of a **12-month Months-On-Hand (MOH)** inventory strategy. It allows supply chain managers to visualize bottlenecks and geospatial risks in real-time.

    ### Key Features
    * **Capacity Heatmap:** Nodes change color based on their utilization. **Red** indicates a capacity breach (Util > 100%).
    * **Geospatial Risk:** View the physical distance and shipping lanes between global suppliers.
    * **Interactive Tunnels:** Hover over links (edges) to see transport lead times and costs.
    
    ### Mathematical Model
    1.  **Inventory Target:** $Target = Monthly Demand \times 12$
    2.  **Utilization:** % of capacity used to maintain the equilibrium of the 12-month buffer.
    3.  **Lead Time Offset:** Calculation of production start dates based on the sum of transport and manufacturing lead times.
    """)
    st.info("💡 **Tip:** Use the slider in the sidebar to stress-test the network. Watch how the US Factory turns red when demand exceeds its max capacity.")

# --- TAB: LOGICAL NETWORK ---
with tab_logic:
    pos = nx.multipartite_layout(network, subset_key="layer")
    
    # Edges
    edge_x, edge_y, mid_x, mid_y, mid_text = [], [], [], [], []
    for s, t, d in network.edges(data=True):
        x0, y0, x1, y1 = pos[s][0], pos[s][1], pos[t][0], pos[t][1]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        mid_x.append((x0 + x1) / 2); mid_y.append((y0 + y1) / 2)
        mid_text.append(f"Route: {s} → {t}<br>Time: {d['lead_time']} days<br>Cost: ${d['cost']}")

    fig_logical = go.Figure()
    fig_logical.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#bbb'), mode='lines', hoverinfo='none'))
    fig_logical.add_trace(go.Scatter(x=mid_x, y=mid_y, mode='markers', marker=dict(size=10, color='rgba(0,0,0,0)'), text=mid_text, hoverinfo='text'))
    
    # Nodes
    nx, ny, nutil, ntext = [], [], [], []
    for n, d in network.nodes(data=True):
        nx.append(pos[n][0]); ny.append(pos[n][1])
        u = getattr(d['obj'], 'peak_utilization', 0)
        nutil.append(u); ntext.append(f"Node: {n}<br>Util: {u:.1f}%")

    fig_logical.add_trace(go.Scatter(x=nx, y=ny, mode='markers', marker=dict(showscale=True, colorscale='RdYlGn_r', color=nutil, size=25, colorbar=dict(title="% Util")), text=ntext, hoverinfo='text'))
    fig_logical.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    st.plotly_chart(fig_logical, use_container_width=True)

# --- TAB: GLOBAL MAP ---
with tab_geo:
    fig_map = go.Figure()
    for s, t, d in network.edges(data=True):
        s_obj, t_obj = network.nodes[s]['obj'], network.nodes[t]['obj']
        fig_map.add_trace(go.Scattergeo(lat=[s_obj.lat, t_obj.lat], lon=[s_obj.lon, t_obj.lon], mode='lines', line=dict(width=1, color='gray'), opacity=0.3, hoverinfo='none'))
        fig_map.add_trace(go.Scattergeo(lat=[(s_obj.lat + t_obj.lat)/2], lon=[(s_obj.lon + t_obj.lon)/2], mode='markers', marker=dict(size=8, color='rgba(0,0,0,0)'), text=f"{s}→{t}: {d['lead_time']}d", hoverinfo='text'))

    mlat, mlon, mutil, mhover = [], [], [], []
    for n, d in network.nodes(data=True):
        mlat.append(d['obj'].lat); mlon.append(d['obj'].lon)
        u = getattr(d['obj'], 'peak_utilization', 0)
        mutil.append(u); mhover.append(f"{n} ({u:.1f}% Util)")

    fig_map.add_trace(go.Scattergeo(lat=mlat, lon=mlon, mode='markers', marker=dict(size=12, color=mutil, colorscale='Reds', showscale=True), text=mhover, hoverinfo='text'))
    fig_map.update_geos(projection_type="natural earth", showcountries=True)
    st.plotly_chart(fig_map, use_container_width=True)

# --- TAB: RAW DATA EXPLORER ---
with tab_data:
    st.header("Source Data Audit")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Nodes (Facilities)")
        st.dataframe(nodes_df, use_container_width=True)
    with col2:
        st.subheader("Edges (Logistics)")
        st.dataframe(edges_df, use_container_width=True)
    
    st.download_button("Download Nodes CSV", nodes_df.to_csv(index=False), "nodes.csv", "text/csv")
