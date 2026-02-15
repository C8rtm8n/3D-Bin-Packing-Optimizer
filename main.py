import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from py3dbp import Packer, Bin, Item
import numpy as np

# --- KONFIGURACE ---
st.set_page_config(page_title="Optimalizace nakládky s hlídáním stability", layout="wide")


# --- MODUL FYZIKÁLNÍ VALIDACE ---
def is_supported(new_item, packed_items):
    """Kontroluje, zda má předmět pod sebou dostatečnou oporu (min 60% plochy)"""
    if float(new_item.position[2]) == 0:
        return True

    x1, y1, z1 = [float(p) for p in new_item.position]
    dx1, dy1, dz1 = [float(d) for d in new_item.get_dimension()]

    support_area = 0
    total_area = dx1 * dy1

    for pi in packed_items:
        px, py, pz = [float(p) for p in pi.position]
        pdx, pdy, pdz = [float(d) for d in pi.get_dimension()]

        # Kontrola, zda je pi přímo pod new_item
        if abs((pz + pdz) - z1) < 0.1:
            ix = max(0, min(x1 + dx1, px + pdx) - max(x1, px))
            iy = max(0, min(y1 + dy1, py + pdy) - max(y1, py))
            support_area += (ix * iy)

    return (support_area / total_area) >= 0.6


# --- VIZUALIZAČNÍ FUNKCE ---
def add_box_trace(fig, pos, dim, name, color, offset_x, offset_y):
    x, y, z = [float(v) for v in pos]
    dx, dy, dz = [float(v) for v in dim]

    fig.add_trace(go.Mesh3d(
        x=[x + offset_x, x + offset_x, x + dx + offset_x, x + dx + offset_x, x + offset_x, x + offset_x,
           x + dx + offset_x, x + dx + offset_x],
        y=[y + offset_y, y + dy + offset_y, y + dy + offset_y, y + offset_y, y + offset_y, y + dy + offset_y,
           y + dy + offset_y, y + offset_y],
        z=[z, z, z, z, z + dz, z + dz, z + dz, z + dz],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.8,
        color=color,
        name=name,
        showlegend=False,
        flatshading=True
    ))


# --- UI APLIKACE ---
st.title("🚛 Optimalizace nakládky s hlídáním stability")
st.markdown("Demo nástroje pro 3D Bin Packing s ohledem na fyzikální stabilitu a těžiště.")

st.sidebar.header("⚙️ Parametry nádoby")
container_type = st.sidebar.selectbox("Typ kontejneru",
                                      ["20' Standard (590x235x239)", "40' HC (1203x235x269)", "Vlastní"])

if container_type == "20' Standard (590x235x239)":
    L, W, H, MW = 590, 235, 239, 22000
elif container_type == "40' HC (1203x235x269)":
    L, W, H, MW = 1203, 235, 269, 28000
else:
    L = st.sidebar.number_input("Délka (cm)", value=600)
    W = st.sidebar.number_input("Šířka (cm)", value=240)
    H = st.sidebar.number_input("Výška (cm)", value=240)
    MW = st.sidebar.number_input("Nosnost (kg)", value=20000)

st.subheader("📋 Seznam položek k nakládce")
col_input1, col_input2 = st.columns([2, 1])

with col_input2:
    uploaded_file = st.file_uploader("Import z Excelu (.xlsx)", type=["xlsx"])
    st.info("💡 Tip: Sloupce musí být: Název, D, Š, V, Kg, Ks")

with col_input1:
    default_data = pd.DataFrame({
        'Název': ['Rozvaděč R1', 'Baterie LiFePo', 'Kabelový buben', 'Chladící jednotka'],
        'D': [120, 20, 150, 80],
        'Š': [100, 100, 150, 80],
        'V': [30, 120, 100, 120],
        'Kg': [450, 300, 600, 150],
        'Ks': [50, 35, 5, 12]
    })

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        df = st.data_editor(default_data, num_rows="dynamic")

# --- VÝPOČETNÍ JÁDRO ---
if st.button("🚀 Spočítat optimální nakládku"):
    # Příprava dat (Master List)
    master_item_list = []
    for _, row in df.iterrows():
        for i in range(int(row['Ks'])):
            master_item_list.append({
                'name': f"{row['Název']} #{i + 1}",
                'dims': (row['D'], row['Š'], row['V']),
                'weight': row['Kg']
            })

    # Stabilní řazení od největší podstavy
    master_item_list.sort(key=lambda x: x['dims'][0] * x['dims'][1], reverse=True)

    final_bins = []
    bin_idx = 0

    # Hlavní balící cyklus
    while len(master_item_list) > 0 and bin_idx < 20:
        packer = Packer()
        current_bin = Bin(f'K{bin_idx + 1}', L, W, H, MW)
        packer.add_bin(current_bin)

        for item_data in master_item_list:
            packer.add_item(Item(item_data['name'], *item_data['dims'], item_data['weight']))

        packer.pack(bigger_first=True)

        bin_res = packer.bins[0]
        valid_items_in_this_bin = []
        names_to_remove = []

        for item in bin_res.items:
            if is_supported(item, valid_items_in_this_bin):
                valid_items_in_this_bin.append(item)
                names_to_remove.append(item.name)

        if not valid_items_in_this_bin:
            # Pokud se už nic nevejde ani do nového prázdného binu, skonči (např. příliš velká krabice)
            break

        # Odstranění úspěšně zabalených položek
        master_item_list = [it for it in master_item_list if it['name'] not in names_to_remove]

        # Těžiště
        total_w = sum([float(i.weight) for i in valid_items_in_this_bin])
        cg = (0, 0, 0)
        if total_w > 0:
            cg = (
                sum([float(i.weight) * (float(i.position[0]) + float(i.width) / 2) for i in
                     valid_items_in_this_bin]) / total_w,
                sum([float(i.weight) * (float(i.position[1]) + float(i.height) / 2) for i in
                     valid_items_in_this_bin]) / total_w,
                sum([float(i.weight) * (float(i.position[2]) + float(i.depth) / 2) for i in
                     valid_items_in_this_bin]) / total_w
            )

        final_bins.append({'items': valid_items_in_this_bin, 'mass': total_w, 'cg': cg})
        bin_idx += 1

    # --- VIZUALIZACE ---
    st.divider()
    if len(master_item_list) > 0:
        st.warning(f"⚠️ {len(master_item_list)} položek se nepodařilo umístit do dostupných kontejnerů.")

    st.subheader("📦 Vizualizace a report")
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for b_idx, b in enumerate(final_bins):
        # Offset pro zobrazení více kontejnerů vedle sebe
        ox, oy = (b_idx % 2) * (L + 150), (b_idx // 2) * (W + 150)

        dev_x = ((b['cg'][0] - L / 2) / L) * 100
        dev_y = ((b['cg'][1] - W / 2) / W) * 100

        with st.expander(f"Kontejner {b_idx + 1} - Podrobnosti", expanded=(b_idx == 0)):
            col_a, col_b = st.columns(2)
            col_a.write(f"**Hmotnost:** {b['mass']} kg")
            col_a.write(f"**Položek:** {len(b['items'])}")
            col_b.write(f"**Těžiště X:** {dev_x:.1f}% od středu")
            col_b.write(f"**Těžiště Y:** {dev_y:.1f}% od středu")

        # Rám kontejneru
        fig.add_trace(go.Scatter3d(
            x=[ox, ox + L, ox + L, ox, ox, ox, ox + L, ox + L, ox, ox, ox, ox + L, ox + L, ox + L, ox + L, ox],
            y=[oy, oy, oy + W, oy + W, oy, oy, oy, oy + W, oy + W, oy, oy, oy + W, oy + W, oy, oy, oy + W],
            z=[0, 0, 0, 0, 0, H, H, H, H, H, 0, 0, H, H, 0, 0],
            mode='lines', line=dict(color='black', width=3), showlegend=False
        ))

        # Položky
        for i_idx, item in enumerate(b['items']):
            add_box_trace(fig, item.position, item.get_dimension(), item.name, colors[i_idx % len(colors)], ox, oy)

        # Těžiště (Červený diamant)
        fig.add_trace(go.Scatter3d(x=[b['cg'][0] + ox], y=[b['cg'][1] + oy], z=[b['cg'][2]],
                                   mode='markers', marker=dict(size=12, color='red', symbol='diamond'),
                                   name=f"Těžiště K{b_idx + 1}"))

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='Délka (cm)'),
            yaxis=dict(title='Šířka (cm)'),
            zaxis=dict(title='Výška (cm)'),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)
