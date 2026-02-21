import streamlit as st
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

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
    
    n_grav = 3
    grav_centers = np.random.uniform(0, width, (n_grav, 2))
    
    N_oversample = N_gen * 5
    x_tmp = np.random.uniform(0, width, N_oversample)
    y_tmp = np.random.uniform(0, height, N_oversample)
    
    dist_all = np.array([np.sqrt((x_tmp - cx)**2 + (y_tmp - cy)**2) for cx, cy in grav_centers])
    min_dists = dist_all.min(axis=0)
    
    weights = np.exp(-min_dists**2 / (2 * 400**2)) 
    weights = weights ** (1 / max(params['grav_str'], 0.1))
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
    
    shape_k = 5.0
    heights = np.clip(np.random.gamma(shape=shape_k, scale=params['scale']/(shape_k-1), size=N_final), min_height, max_height)
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

# --- 3. FELHASZNÁLÓI FELÜLET ÉS FŐ LOGIKA ---
with st.sidebar:
    st.header("⚙️ Beállítások")
    in_intensity = st.slider("Cél sűrűség (db/m²)", 0.00005, 0.005, 0.0020, step=0.00005, format="%.5f")
    in_scale = st.slider("Magasság scale (módusz)", 5, 50, 15)
    in_grav_str = st.slider("Sűrűsödési erő", 0, 10, 3)
    in_chewed = st.slider("Valódi rágottság (%)", 0, 100, 30)
    in_runs = st.slider("Szimulációs futások száma", 2, 100, 5)

    if 'KTT' not in st.session_state: st.session_state['KTT'] = 20
    if 'Gy' not in st.session_state: st.session_state['Gy'] = 20
    if 'MJ' not in st.session_state: st.session_state['MJ'] = 20
    if 'MCs' not in st.session_state: st.session_state['MCs'] = 20

    def sync_sliders(changed_key):
        current_total = st.session_state['KTT'] + st.session_state['Gy'] + st.session_state['MJ'] + st.session_state['MCs']
        if current_total > 100:
            excess = current_total - 100
            others = [k for k in ['KTT', 'Gy', 'MJ', 'MCs'] if k != changed_key]
            for k in others:
                if st.session_state[k] >= excess:
                    st.session_state[k] -= excess
                    excess = 0
                    break
                else:
                    excess -= st.session_state[k]
                    st.session_state[k] = 0

    p_ktt = st.slider("KTT (%)", 0, 100, key='KTT', on_change=sync_sliders, args=('KTT',))
    p_gy = st.slider("Gy (%)", 0, 100, key='Gy', on_change=sync_sliders, args=('Gy',))
    p_mj = st.slider("MJ (%)", 0, 100, key='MJ', on_change=sync_sliders, args=('MJ',))
    p_mcs = st.slider("MCs (%)", 0, 100, key='MCs', on_change=sync_sliders, args=('MCs',))
    p_babe = max(0, 100 - (p_ktt + p_gy + p_mj + p_mcs))
    st.info(f"BaBe (maradék): {p_babe}%")

if st.button("SZIMULÁCIÓ FUTTATÁSA", use_container_width=True):
    raw_probs = np.array([p_ktt, p_gy, p_mj, p_mcs, p_babe], dtype=float)
    corrected_probs = raw_probs / raw_probs.sum()

    sim_params = {
        'intensity': in_intensity, 'scale': in_scale, 'grav_str': in_grav_str,
        'chewed_p': in_chewed,
        'sp_names': ['KTT', 'Gy', 'MJ', 'MCs', 'BaBe'],
        'sp_probs': corrected_probs 
    }

    first_run_stats = {}
    first_df = None
    my_bar = st.progress(0, text="Szimulációk futtatása...")

    for i in range(in_runs):
        current_df = run_forest_simulation(sim_params)
        if i == 0:
            first_df = current_df
            t_df_f = current_df[current_df['T'] == 1]
            c_df_f = current_df[current_df['C'] == 1]
            c_large_f = c_df_f[c_df_f['height'] > 50]
            c_small_f = c_df_f[c_df_f['height'] <= 50]
            c_dens_f = (len(c_large_f) / area_big_circle) + (len(c_small_f) / area_small_circles) if area_big_circle > 0 else 0
            
            first_run_stats = {
                'S_count': len(current_df),
                'T_count': len(t_df_f),
                'C_count': len(c_df_f),
                'S_density': len(current_df) / (width * height),
                'T_density': (t_df_f['height'].apply(lambda h: 1/h).sum() / width) if len(t_df_f) > 0 else 0,
                'C_density': c_dens_f,
                'S_chewed': current_df['chewed'].mean() * 100,
                'T_chewed': t_df_f['chewed'].mean() * 100 if len(t_df_f) > 0 else 0,
                'C_chewed': c_df_f['chewed'].mean() * 100 if len(c_df_f) > 0 else 0
            }
        my_bar.progress((i + 1) / in_runs)

    my_bar.empty()

    # --- TÁBLÁZAT MEGJELENÍTÉSE ---
    summary_table = {
        "Paraméter": ["Darabszám (count)", "Sűrűség (density)", "Rágottság (chewed_%)"],
        "Szimuláció (S)": [
            f"{first_run_stats['S_count']} db", 
            f"{first_run_stats['S_density']:.5f}", 
            f"{first_run_stats['S_chewed']:.1f}%"
        ],
        "Transzekt (T)": [
            f"{first_run_stats['T_count']} db", 
            f"{first_run_stats['T_density']:.5f}", 
            f"{first_run_stats['T_chewed']:.1f}%"
        ],
        "Mintakör (C)": [
            f"{first_run_stats['C_count']} db", 
            f"{first_run_stats['C_density']:.5f}", 
            f"{first_run_stats['C_chewed']:.1f}%"
        ]
    }
    
    st.subheader("📊 Az első futás részletes eredményei")
    st.table(pd.DataFrame(summary_table))

    df = first_df 

    st.markdown("---")
    st.subheader("🌲 A szimulált erdő fafaj-összetétele (Első futás)")
    st.markdown(
        f"""
        <div style="display: flex; height: 35px; width: 100%; border-radius: 8px; overflow: hidden; border: 2px solid #ddd; margin-bottom: 20px;">
            <div style="width: {p_ktt}%; background-color: {species_colors['KTT']}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{p_ktt if p_ktt > 5 else ''}%</div>
            <div style="width: {p_gy}%; background-color: {species_colors['Gy']}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{p_gy if p_gy > 5 else ''}%</div>
            <div style="width: {p_mj}%; background-color: {species_colors['MJ']}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{p_mj if p_mj > 5 else ''}%</div>
            <div style="width: {p_mcs}%; background-color: {species_colors['MCs']}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{p_mcs if p_mcs > 5 else ''}%</div>
            <div style="width: {p_babe}%; background-color: {species_colors['BaBe']}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{p_babe if p_babe > 5 else ''}%</div>
        </div>
        """, unsafe_allow_html=True
    )

    st.subheader("📊 Magasság eloszlás")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
    sns.histplot(df['height'], kde=True, bins=30, color="forestgreen", ax=ax_dist, stat="density")
    st.pyplot(fig_dist)
    plt.close(fig_dist)
    
    st.subheader("🧊 3D Nézet")
    fig_3d = plt.figure(figsize=(10, 7))
    ax3d = fig_3d.add_subplot(111, projection='3d')
    for sp in sim_params['sp_names']:
        sp_df = df[df['species'] == sp]
        if not sp_df.empty:
            ax3d.scatter(sp_df['X'], sp_df['Y'], sp_df['height'], color=species_colors[sp], s=sp_df['height']*2, alpha=0.7, label=sp)
    st.pyplot(fig_3d)
    plt.close(fig_3d)

    st.subheader("🗺️ Transzekt mintavétel felülnézetből")
    fig_map, ax_map = plt.subplots(figsize=(10, 10))
    ax_map.scatter(df['X'], df['Y'], c='lightgrey', s=5, alpha=0.3, label='Erdő egyedei')
    t_df = df[df['T'] == 1]
    if not t_df.empty:
        for sp in sim_params['sp_names']:
            sp_t = t_df[t_df['species'] == sp]
            if not sp_t.empty:
                ax_map.scatter(sp_t['X'], sp_t['Y'], color=species_colors[sp], s=20, label=f'{sp} (mintában)')
    ax_map.plot([0, width], [0, height], color='red', linestyle='--', linewidth=1, label='Transzekt tengely')
    ax_map.set_xlim(0, width)
    ax_map.set_ylim(0, height)
    ax_map.set_aspect('equal')
    st.pyplot(fig_map)
    plt.close(fig_map)
    
    st.markdown("---")
    st.subheader("🦌 Rágottság mértéke fafajonként")
    fig_chew, ax_chew = plt.subplots(figsize=(10, 5))
    species_chewed = df.groupby('species')['chewed'].mean() * 100
    full_species_list = sim_params['sp_names']
    chew_values = [species_chewed.get(sp, 0) for sp in full_species_list]
    colors = [species_colors[sp] for sp in full_species_list]
    ax_chew.bar(full_species_list, chew_values, color=colors, edgecolor='black', alpha=0.8)
    ax_chew.axhline(in_chewed, color='red', linestyle='--', label=f'Cél ({in_chewed}%)')
    ax_chew.set_ylim(0, 110)
    st.pyplot(fig_chew)
    plt.close(fig_chew)

    st.markdown("---")
    st.subheader("🎯 Mintakörös mintavétel felülnézetből")
    fig_circ, ax_circ = plt.subplots(figsize=(10, 10))
    ax_circ.scatter(df['X'], df['Y'], c='lightgray', s=5, alpha=0.3)
    c_df = df[df['C'] == 1]
    if not c_df.empty:
        for sp in sim_params['sp_names']:
            sp_c = c_df[c_df['species'] == sp]
            if not sp_c.empty:
                ax_circ.scatter(sp_c['X'], sp_c['Y'], color=species_colors[sp], s=30)
    
    circle_big_patch = patches.Circle(center_big, r_big, color='navy', fill=False, linestyle='--')
    ax_circ.add_patch(circle_big_patch)
    for cs in centers_small:
        circle_small_patch = patches.Circle(cs, r_small, color='dodgerblue', fill=False, linestyle=':')
        ax_circ.add_patch(circle_small_patch)

    ax_circ.set_xlim(0, width)
    ax_circ.set_ylim(0, height)
    ax_circ.set_aspect('equal')
    st.pyplot(fig_circ)
    plt.close(fig_circ)
