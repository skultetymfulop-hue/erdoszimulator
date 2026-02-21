import streamlit as st
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# --- 1. ALAPBEÁLLÍTÁSOK ---
st.set_page_config(page_title="Profi Erdő Szimulátor", layout="centered")

width, height = 1500, 1500 
max_height = 200
min_height = 3
R_core = 5  
center_big = (width/2, height/2)
r_big = 564
r_small = 126
centers_small = [(width/4, height/4), (3*width/4, height/4), 
                 (width/4, 3*height/4), (3*width/4, 3*height/4)]

area_big_circle = math.pi * (r_big**2)
area_small_circles = 4 * (math.pi * (r_small**2))

species_colors = {
    'KTT': '#1f77b4', 'Gy': '#2ca02c', 'MJ': '#ff7f0e', 'MCs': '#d62728', 'BaBe': '#9467bd'
}

def point_line_distance(x, y, x1, y1, x2, y2):
    num = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1))
    den = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return num / den

def get_weighted_height_mode(df_subset, is_transzekt=False):
    if len(df_subset) == 0: return 0
    rounded_heights = df_subset['height'].round()
    if is_transzekt:
        weights = 1 / df_subset['height']
        counts = {}
        for h, w in zip(rounded_heights, weights):
            counts[h] = counts.get(h, 0) + w
        return int(max(counts, key=counts.get))
    else:
        mode_result = stats.mode(rounded_heights, keepdims=True)
        return int(mode_result.mode[0])

# --- 2. SZIMULÁCIÓS FÜGGVÉNY ---
def run_forest_simulation(params):
    target_intensity = params['intensity']
    expected_n = int(target_intensity * width * height * 1.5)
    N_gen = np.random.poisson(expected_n)
    
    # Csoportosulás (Gravitációs központok)
    n_grav = 3
    grav_centers = np.random.uniform(0, width, (n_grav, 2))
    
    N_oversample = N_gen * 5
    x_tmp = np.random.uniform(0, width, N_oversample)
    y_tmp = np.random.uniform(0, height, N_oversample)
    
    dist_all = np.array([np.sqrt((x_tmp - cx)**2 + (y_tmp - cy)**2) for cx, cy in grav_centers])
    min_dists = dist_all.min(axis=0)
    
    # Itt használjuk a sűrűsödési erőt (grav_str)
    weights = np.exp(-min_dists**2 / (2 * 400**2)) 
    weights = weights ** (1 / max(params['grav_str'], 0.1))
    weights /= weights.max()
    
    keep_mask = np.random.uniform(0, 1, N_oversample) < weights
    accepted = np.column_stack((x_tmp, y_tmp))[keep_mask]
    
    if len(accepted) > N_gen:
        accepted = accepted[np.random.choice(len(accepted), N_gen, replace=False)]
    
    # Ritkítás (R_core távolság)
    final_keep = np.ones(len(accepted), dtype=bool)
    R_sq = R_core**2
    for i in range(len(accepted)):
        if not final_keep[i]: continue
        d_sq = np.sum((accepted[i] - accepted)**2, axis=1)
        final_keep[(d_sq < R_sq) & (d_sq > 0)] = False
    
    final_coords = accepted[final_keep]
    N_final = len(final_coords)
    
    shape_k = 2.0
    heights = np.clip(np.random.gamma(shape=shape_k, scale=params['scale']/shape_k, size=N_final), min_height, max_height)
    fajok = np.random.choice(params['sp_names'], size=N_final, p=params['sp_probs'])
    ragottsag = np.random.uniform(0, 100, size=N_final) < params['chewed_p']
    
    results = []
    for i in range(N_final):
        x, y, h = final_coords[i,0], final_coords[i,1], heights[i]
        in_t = 1 if point_line_distance(x, y, 0, 0, width, height) <= h else 0
        in_c = 0
        if h > 50 and math.dist((x, y), center_big) <= r_big: in_c = 1
        elif h <= 50:
            for cs in centers_small:
                if math.dist((x, y), cs) <= r_small: in_c = 1; break
        
        results.append({
            "X": x, "Y": y, "height": h, "species": fajok[i], 
            "chewed": int(ragottsag[i]), "T": in_t, "C": in_c
        })
    return pd.DataFrame(results)

# --- 3. FELHASZNÁLÓI FELÜLET ---
with st.sidebar:
    st.header("⚙️ Beállítások")
    in_intensity = st.slider("Cél sűrűség (db/m²)", 0.0005, 0.0100, 0.0020, step=0.0005, format="%.4f")
    in_scale = st.slider("Magasság scale (módusz)", 5, 50, 15)
    in_grav_str = st.slider("Sűrűsödési erő", 0, 10, 3)
    in_chewed = st.slider("Valódi rágottság (%)", 0, 100, 30)
    
    st.markdown("---")
   st.subheader("🌿 Fajösszetétel (Dinamikus korláttal)")

    # 1. KTT csúszka - ez az alap, 0-100 között bármi lehet
    p_ktt = st.sidebar.slider("KTT (%)", 0, 100, 20, key="s_ktt")

    # 2. Gy csúszka - a max értéke a maradék (100 - KTT)
    max_gy = 100 - p_ktt
    p_gy = st.sidebar.slider("Gy (%)", 0, max_gy, min(20, max_gy), key="s_gy")

    # 3. MJ csúszka - a max értéke a maradék (100 - KTT - Gy)
    max_mj = 100 - p_ktt - p_gy
    p_mj = st.sidebar.slider("MJ (%)", 0, max_mj, min(20, max_mj), key="s_mj")

    # 4. MCs csúszka - a max értéke a maradék (100 - KTT - Gy - MJ)
    max_mcs = 100 - p_ktt - p_gy - p_mj
    p_mcs = st.sidebar.slider("MCs (%)", 0, max_mcs, min(20, max_mcs), key="s_mcs")

    # 5. BaBe - a végső maradék automatikusan
    p_babe = 100 - p_ktt - p_gy - p_mj - p_mcs
    
    # Esztétikus visszajelzés
    st.info(f"Fajok összege: {p_ktt + p_gy + p_mj + p_mcs + p_babe}%")
    st.success(f"Automatikus BaBe maradék: {p_babe}%")

if st.button("SZIMULÁCIÓ FUTTATÁSA", use_container_width=True):
    # Normalizálás a biztonság kedvéért
    raw_probs = np.array([p_ktt, p_gy, p_mj, p_mcs, p_babe], dtype=float)
    corrected_probs = raw_probs / raw_probs.sum()

    sim_params = {
        'intensity': in_intensity, 'scale': in_scale, 'grav_str': in_grav_str,
        'chewed_p': in_chewed,
        'sp_names': ['KTT', 'Gy', 'MJ', 'MCs', 'BaBe'],
        'sp_probs': corrected_probs 
    }
    
    df = run_forest_simulation(sim_params)
    
    if not df.empty:
        # --- STATISZTIKA ---
        t_df = df[df['T'] == 1]
        c_df = df[df['C'] == 1]
        
        stats_data = {
            "Paraméter": ["Egyedszám", "Sűrűség (db/m²)", "Scale (H)", "Rágottság"],
            "Valódi (S)": [len(df), f"{len(df)/(width*height):.4f}", get_weighted_height_mode(df), f"{df['chewed'].mean()*100:.1f}%"],
            "Transzekt (T)": [len(t_df), f"{(t_df['height'].apply(lambda h: 1/h).sum()/width) if len(t_df)>0 else 0:.4f}", get_weighted_height_mode(t_df, True), f"{t_df['chewed'].mean()*100 if len(t_df)>0 else 0:.1f}%"],
            "Mintakör (C)": [len(c_df), "N/A", get_weighted_height_mode(c_df), f"{c_df['chewed'].mean()*100 if len(c_df)>0 else 0:.1f}%"]
        }
        st.subheader("📊 Becslési eredmények")
        st.table(pd.DataFrame(stats_data))

        # --- 3D ÁBRA ---
        st.subheader("🧊 Az erdő 3D nézete (Fajok szerint)")
        fig_3d = plt.figure(figsize=(10, 7))
        ax3d = fig_3d.add_subplot(111, projection='3d')
        for sp in sim_params['sp_names']:
            sp_df = df[df['species'] == sp]
            if not sp_df.empty:
                ax3d.scatter(sp_df['X'], sp_df['Y'], sp_df['height'], color=species_colors[sp], s=sp_df['height']*2, alpha=0.7, label=sp)
                for _, tree in sp_df.iterrows():
                    ax3d.plot([tree['X'], tree['X']], [tree['Y'], tree['Y']], [0, tree['height']], color='brown', alpha=0.1, linewidth=0.5)
        ax3d.set_zlim(0, max_height)
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Magasság (m)")
        ax3d.legend()
        st.pyplot(fig_3d)
        plt.close(fig_3d)

