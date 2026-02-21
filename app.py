import streamlit as st
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats

# --- 1. ALAPBEÁLLÍTÁSOK ---
st.set_page_config(page_title="Profi Erdő Szimulátor", layout="centered")

width, height = 1500, 1500 
area_ha = (width * height) / 10000 
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

# --- 2. KORREKCIÓS MOTOR ---
def get_retention_ratio(intensity, R_core):
    test_n = int(intensity * width * height)
    if test_n < 10: return 1.0
    coords = np.random.uniform(0, width, (min(test_n, 1000), 2))
    R_sq = R_core**2
    keep = np.ones(len(coords), dtype=bool)
    for i in range(len(coords)):
        if not keep[i]: continue
        dists = np.sum((coords[i] - coords)**2, axis=1)
        if np.any((dists < R_sq) & (dists > 0)): keep[i] = False
    return np.sum(keep) / len(coords)

# --- 3. SZIMULÁCIÓS FÜGGVÉNY ---
def run_forest_simulation(params):
    target_intensity = params['intensity']
    retention = get_retention_ratio(target_intensity, R_core)
    corrected_intensity = target_intensity / max(retention, 0.1)
    
    expected_n = int(corrected_intensity * width * height)
    N_gen = np.random.poisson(expected_n)
    grav_centers = np.random.uniform(0, width, (params['n_grav'], 2))
    
    N_oversample = N_gen * 5
    x_tmp = np.random.uniform(0, width, N_oversample)
    y_tmp = np.random.uniform(0, height, N_oversample)
    
    dist_all = np.array([np.sqrt((x_tmp - cx)**2 + (y_tmp - cy)**2) for cx, cy in grav_centers])
    min_dists = dist_all.min(axis=0)
    weights = np.exp(-min_dists**2 / (2 * params['sigma']**2))
    weights /= weights.max()
    
    keep_mask = np.random.uniform(0, 1, N_oversample) < weights
    accepted = np.column_stack((x_tmp, y_tmp))[keep_mask]
    
    if len(accepted) > N_gen:
        accepted = accepted[np.random.choice(len(accepted), N_gen, replace=False)]
    
    final_keep = np.ones(len(accepted), dtype=bool)
    R_sq = R_core**2
    for i in range(len(accepted)):
        if not final_keep[i]: continue
        d_sq = np.sum((accepted[i] - accepted)**2, axis=1)
        final_keep[(d_sq < R_sq) & (d_sq > 0)] = False
    
    final_coords = accepted[final_keep]
    N_final = len(final_coords)
    
    dist_final = np.array([np.sqrt((final_coords[:,0] - cx)**2 + (final_coords[:,1] - cy)**2) for cx, cy in grav_centers]).min(axis=0)
    shape_k = 2.0
    base_h = min_height + np.random.gamma(shape=shape_k, scale=params['scale']/shape_k, size=N_final)
    gauss_eff = np.exp(-0.5 * (dist_final / params['sigma_h'])**2)
    heights = np.clip(base_h * (1 + params['grav_str'] * gauss_eff), min_height, max_height)
    
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
                if math.dist((x, y), cs) <= r_small:
                    in_c = 1; break
        
        results.append({
            "X": x, "Y": y, "height": h, "species": fajok[i], 
            "chewed": int(ragottsag[i]), "T": in_t, "C": in_c
        })
    return pd.DataFrame(results)

# --- 4. FELHASZNÁLÓI FELÜLET ---
st.title("🌲 Erdő Szimulátor és Becslés Validátor")

with st.sidebar:
    st.header("⚙️ Alapbeállítások")
    in_intensity = st.slider("Cél sűrűség (db/m²)", 0.0005, 0.0100, 0.0020, step=0.0005, format="%.4f")
    in_scale = st.slider("Magasság scale (módusz)", 5, 50, 15)
    in_grav_str = st.slider("Sűrűsödési erő", 0, 10, 3)
    in_chewed = st.slider("Valódi rágottság (%)", 0, 100, 30)

    st.markdown("---")
    st.subheader("🌿 Fajösszetétel")
    p_ktt = st.slider("KTT (%)", 0, 100, 20)
    rem1 = 100 - p_ktt
    p_gy = st.slider("Gy (%)", 0, rem1, min(20, rem1))
    rem2 = rem1 - p_gy
    p_mj = st.slider("MJ (%)", 0, rem2, min(20, rem2))
    rem3 = rem2 - p_mj
    p_mcs = st.slider("MCs (%)", 0, rem3, min(20, rem3))
    p_babe = rem3 - p_mcs
    st.info(f"BaBe: {p_babe}%")

if st.button("SZIMULÁCIÓ ÉS BECSLÉS FUTTATÁSA", use_container_width=True):
    sim_params = {
        'intensity': in_intensity, 'scale': in_scale, 'grav_str': in_grav_str,
        'chewed_p': in_chewed, 'n_grav': 3, 'sigma': 400, 'sigma_h': 50.0,
        'sp_names': ['KTT', 'Gy', 'MJ', 'MCs', 'BaBe'],
        'sp_probs': [p_ktt/100, p_gy/100, p_mj/100, p_mcs/100, p_babe/100]
    }
    
    df = run_forest_simulation(sim_params)
    
    if not df.empty:
        # --- STATISZTIKAI SZÁMÍTÁSOK ---
        t_df = df[df['T'] == 1]
        c_df = df[df['C'] == 1]

        # 1. Darabszámok (Counts)
        s_count = len(df)
        t_count = len(t_df)
        c_count = len(c_df)

        # 2. Rágottság becslése
        s_chewed = df['chewed'].mean() * 100
        t_chewed = t_df['chewed'].mean() * 100 if t_count > 0 else 0
        c_chewed = c_df['chewed'].mean() * 100 if c_count > 0 else 0

        # 3. Scale (Magasság módusz)
        s_
      
