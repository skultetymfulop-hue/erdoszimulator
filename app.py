import streamlit as st
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

# --- 1. KONFIGURÁCIÓ ÉS SEGÉDFÜGGVÉNYEK ---
st.set_page_config(page_title="Erdő Szimulátor", layout="centered")

# Globális állandók
width, height = 1500, 1500
max_height = 200
h_min = 3
R_core = 5
n_gravity_points = 3
gravity_strength = 3
sigma = 400
sigma_height = 50.0

# Mintakörök geometriája
center_big = (width/2, height/2)
r_big = 564
r_small = 126
centers_small = [(width/4, height/4), (3*width/4, height/4), 
                 (width/4, 3*height/4), (3*width/4, 3*height/4)]

def point_line_distance(x, y, x1, y1, x2, y2):
    num = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1))
    den = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return num / den

# --- 2. A MOTOR (Szimulációs függvény) ---
def run_simulation(params):
    scale = params['scale']
    target_intensity = params['intensity'] 
    clumping = params.get('clumping_factor', 0.5)
    species = params['species']
    species_probs = params['species_probs']
    chewed_probs_map = params['chewed_probs']

    expected_trees = target_intensity * width * height
    N_trees = np.random.poisson(expected_trees)
    
    if N_trees == 0: return pd.DataFrame(), np.array([])

    gravity_centers = np.random.uniform(0, width, (n_gravity_points, 2))

    # Pontok generálása
    if clumping == 0:
        x_coords = np.random.uniform(0, width, N_trees)
        y_coords = np.random.uniform(0, height, N_trees)
    else:
        N_oversample = int(max(N_trees * 5, 1000)) 
        x_temp = np.random.uniform(0, width, N_oversample)
        y_temp = np.random.uniform(0, height, N_oversample)
        dist_all = np.array([np.sqrt((x_temp - cx)**2 + (y_temp - cy)**2) for cx, cy in gravity_centers])
        dist_min = dist_all.min(axis=0)
        weights = np.exp(-dist_min**2 / (2 * sigma**2))
        normalized_weights = weights / np.max(weights)
        keep_mask = np.random.uniform(0, 1, N_oversample) > (clumping * (1 - normalized_weights))
        accepted = np.column_stack((x_temp, y_temp))[keep_mask]
        if len(accepted) > N_trees:
            accepted = accepted[np.random.choice(len(accepted), N_trees, replace=False)]
        x_coords, y_coords = accepted[:, 0], accepted[:, 1]

    # Magasság és Fafaj
    N_final = len(x_coords)
    dist_all_final = np.array([np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2) for cx, cy in gravity_centers])
    dist_min_final = dist_all_final.min(axis=0)
    base_h = np.random.exponential(scale=scale, size=N_final)
    gauss_eff = np.exp(-0.5 * (dist_min_final / sigma_height)**2)
    heights = np.clip(base_h * (1 + gravity_strength * gauss_eff), h_min, max_height)
    
    assigned_species = np.random.choice(species, size=N_final, p=species_probs)
    chewed_probs = np.array([chewed_probs_map.get(s, 0) / 100 for s in assigned_species])
    is_chewed = np.random.uniform(0, 1, size=N_final) < chewed_probs

    # Matern II szűrés és mintavétel (leegyszerűsítve az apphoz)
    trees_df = pd.DataFrame({'X': x_coords, 'Y': y_coords, 'H': heights, 'F': assigned_species, 'R': is_chewed})
    
    sampled = []
    for _, tree in trees_df.iterrows():
        d_tr = point_line_distance(tree['X'], tree['Y'], 0, 0, width, height)
        in_tr = 1 if d_tr <= tree['H'] else 0
        in_c = 0
        dist_c = math.dist((tree['X'], tree['Y']), center_big)
        if tree['H'] > 50 and dist_c <= r_big: in_c = 1
        elif tree['H'] <= 50:
            for cs in centers_small:
                if math.dist((tree['X'], tree['Y']), cs) <= r_small:
                    in_c = 1; break
        
        sampled.append({
            "X": tree['X'], "Y": tree['Y'], "Transzekt": in_tr,
            "Mintakör": in_c, "Magasság": tree['H'], "Rágottság": int(tree['R']), "Faj": tree['F']
        })
    return pd.DataFrame(sampled), gravity_centers

# --- 3. FELHASZNÁLÓI FELÜLET (UI) ---
st.title("🌲 Interaktív Erdő Szimulátor")
st.write("Állítsd be a paramétereket a telefonodon!")

# Paraméterek bekérése
input_intensity = st.slider("Fa sűrűség", 0.0005, 0.0100, 0.0020, step=0.0005, format="%.4f")
input_scale = st.slider("Átlagos magasság (scale)", 10, 50, 20)
input_clumping = st.slider("Csoportosulás (clumping)", 0.0, 1.0, 0.5)
input_chewed = st.slider("Rágottsági esély (%)", 0, 100, 70)

if st.button("SZIMULÁCIÓ INDÍTÁSA 🚀", use_container_width=True):
    params = {
        'scale': input_scale,
        'intensity': input_intensity,
        'clumping_factor': input_clumping,
        'species': ['GY'],
        'species_probs': np.array([1.0]),
        'chewed_probs': {'GY': input_chewed}
    }

    with st.spinner('Erdő növesztése...'):
        df, gravs = run_simulation(params)

    if not df.empty:
        # 1. Grafikon: Transzekt nézet
        st.subheader("1. Transzekt mintavétel")
        fig1, ax1 = plt.subplots(figsize=(8,8))
        sns.scatterplot(data=df, x="X", y="Y", hue="Transzekt", palette={0: "gray", 1: "red"}, 
                        size="Magasság", style="Rágottság", markers={0: 'o', 1: '^'}, alpha=0.7, ax=ax1)
        ax1.plot([0, 1500], [0, 1500], color='black', linestyle='--')
        ax1.set_aspect('equal')
        st.pyplot(fig1)

        # 2. Grafikon: Mintakör nézet
        st.subheader("2. Mintakörös mintavétel")
        fig2, ax2 = plt.subplots(figsize=(8,8))
        sns.scatterplot(data=df, x="X", y="Y", hue="Mintakör", palette={0: "lightgray", 1: "#3498db"}, 
                        size="Magasság", style="Rágottság", markers={0: 'o', 1: '^'}, alpha=0.7, ax=ax2)
        # Körök berajzolása
        ax2.add_patch(patches.Circle(center_big, r_big, color='navy', fill=False, linestyle='--'))
        for cs in centers_small:
            ax2.add_patch(patches.Circle(cs, r_small, color='dodgerblue', fill=False, linestyle=':'))
        ax2.set_aspect('equal')
        st.pyplot(fig2)

        # 3. Statisztika: Darabszámok
        st.subheader("3. Mérési eredmények")
        t_count = df['Transzekt'].sum()
        c_count = df['Mintakör'].sum()
        st.metric("Transzektben talált fák", f"{int(t_count)} db")
        st.metric("Mintakörökben talált fák", f"{int(c_count)} db")
    else:

        st.error("Nem születtek fák ezzel a beállítással!")

