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
L_transsect = math.sqrt(width**2 + height**2)

species_colors = {
    'KTT': '#1f77b4', 'Gy': '#2ca02c', 'MJ': '#ff7f0e', 'MCs': '#d62728', 'BaBe': '#9467bd'
}

def point_line_distance(x, y, x1, y1, x2, y2):
    num = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1))
    den = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return num / den

# STABILABB: Átlag számítása súlyozva (Transzekthez) vagy anélkül
def get_weighted_height_mean(df_subset, is_transzekt=False):
    if len(df_subset) == 0: return 0
    if is_transzekt:
        # Horvitz-Thompson korrekció az átlaghoz: sum(h * 1/h) / sum(1/h) = n / sum(1/h)
        return len(df_subset) / (1 / df_subset['height']).sum()
    else:
        return df_subset['height'].mean()

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
    
    # --- JAVÍTOTT MAGASSÁG GENERÁLÁS ---
    shape_k = params['shape_k']
    target_mode = params['mode']
    # Scale kiszámítása a móduszból: mode = (shape-1)*theta
    theta = target_mode / (shape_k - 1)
    
    heights = np.random.gamma(shape=shape_k, scale=theta, size=N_final)
    heights = np.clip(heights, min_height, max_height)
    
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
    in_intensity = st.slider("Cél sűrűség (db/cm²)", 0.00005, 0.005, 0.0020, step=0.00005, format="%.5f")
    
    # ÚJ CSÚSZKÁK A KAVARODÁS ELLEN
    in_mode = st.slider("Leggyakoribb magasság (módusz)", 5, 50, 15)
    in_shape = st.slider("Változatosság (alacsonyabb = több óriás fa)", 1.2, 5.0, 2.0)
    
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
        'intensity': in_intensity, 'mode': in_mode, 'shape_k': in_shape, 'grav_str': in_grav_str,
        'chewed_p': in_chewed,
        'sp_names': ['KTT', 'Gy', 'MJ', 'MCs', 'BaBe'],
        'sp_probs': corrected_probs 
    }

    all_runs_errors = []
    first_run_stats = {}
    first_df = None
    my_bar = st.progress(0, text="Szimulációk futtatása...")

    for i in range(in_runs):
        current_df = run_forest_simulation(sim_params)
        t_df = current_df[current_df['T'] == 1]
        c_df = current_df[current_df['C'] == 1]
        
        # 1. VALÓDI ÉRTÉKEK
        s_dens = len(current_df) / (width * height)
        s_height_avg = get_weighted_height_mean(current_df)
        s_chew = current_df['chewed'].mean() * 100

        # 2. TRANSZEKT BECSLÉS
        if len(t_df) > 0:
            t_density = (1 / (2.0 * t_df['height'] * L_transsect)).sum()
            t_height_avg = get_weighted_height_mean(t_df, is_transzekt=True)
            t_chew = t_df['chewed'].mean() * 100
        else:
            t_density = t_height_avg = t_chew = 0.0
        
        # 3. MINTAKÖRÖS BECSLÉS
       # 3. MINTAKÖRÖS BECSLÉS (Javított, rétegzett súlyozással)
        c_small = c_df[c_df['height'] <= 50]
        c_large = c_df[c_df['height'] > 50]
        
        # Sűrűség számítása rétegenként
        d_small = (len(c_small) / area_small_circles) if area_small_circles > 0 else 0
        d_big = (len(c_large) / area_big_circle) if area_big_circle > 0 else 0
        c_dens = d_small + d_big

        # Súlyozott átlagmagasság és rágottság számítása
        if c_dens > 0:
            # Rétegátlagok kiszámítása
            avg_h_small = c_small['height'].mean() if len(c_small) > 0 else 0
            avg_h_large = c_large['height'].mean() if len(c_large) > 0 else 0
            
            avg_chew_small = c_small['chewed'].mean() if len(c_small) > 0 else 0
            avg_chew_large = c_large['chewed'].mean() if len(c_large) > 0 else 0

            # Súlyozás a becsült sűrűségek (D) alapján
            c_height_avg = (d_small * avg_h_small + d_big * avg_h_large) / c_dens
            c_chew = ((d_small * avg_chew_small + d_big * avg_chew_large) / c_dens) * 100
        else:
            c_height_avg = 0
            c_chew = 0

        # MAPE számítás (Módusz helyett Átlagmagasságra a stabilitásért)
        all_runs_errors.append({
            't_err_dens': abs((s_dens - t_density) / s_dens) if s_dens > 0 else 0,
            't_err_height': abs((s_height_avg - t_height_avg) / s_height_avg) if s_height_avg > 0 else 0,
            't_err_chew': abs((s_chew - t_chew) / s_chew) if s_chew > 0 else 0,
            'c_err_dens': abs((s_dens - c_dens) / s_dens) if s_dens > 0 else 0,
            'c_err_height': abs((s_height_avg - c_height_avg) / s_height_avg) if s_height_avg > 0 else 0,
            'c_err_chew': abs((s_chew - c_chew) / s_chew) if s_chew > 0 else 0
        })

        if i == 0:
            first_df = current_df
            first_run_stats = {
                'S_count': len(current_df), 'T_count': len(t_df), 'C_count': len(c_df),
                'S_density': s_dens, 'T_density': t_density, 'C_density': c_dens,
                'S_chewed': s_chew, 'T_chewed': t_chew, 'C_chewed': c_chew
            }
        my_bar.progress((i + 1) / in_runs)

    my_bar.empty()

    # --- TÁBLÁZATOK ---
    errors_df = pd.DataFrame(all_runs_errors)
    mape_table = {
        "Sorok (MAPE)": ["MAPE_density", "MAPE_height_avg", "MAPE_chewed"],
        "Transzekt (T)": [
            f"{errors_df['t_err_dens'].mean()*100:.2f}%", 
            f"{errors_df['t_err_height'].mean()*100:.2f}%", 
            f"{errors_df['t_err_chew'].mean()*100:.2f}%"
        ],
        "Mintakör (C)": [
            f"{errors_df['c_err_dens'].mean()*100:.2f}%", 
            f"{errors_df['c_err_height'].mean()*100:.2f}%", 
            f"{errors_df['c_err_chew'].mean()*100:.2f}%"
        ]
    }
    st.subheader(f"📈 MAPE eredmények ({in_runs} futás alapján)")
    st.table(pd.DataFrame(mape_table))
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

    # (A többi vizualizációs kódod változatlan marad...)
    df = first_df
   # --- ELEGYARÁNY VIZUALIZÁCIÓ (S, T, C) ---
    st.markdown("---")
    st.subheader("🌲 Fafaj-összetétel összehasonlítása (Első futás)")

    def get_species_percentages(dataframe, species_list):
        if len(dataframe) == 0:
            return [0] * len(species_list)
        counts = dataframe['species'].value_counts(normalize=True) * 100
        return [counts.get(sp, 0) for sp in species_list]

    # Százalékok kiszámítása
    sp_list = ['KTT', 'Gy', 'MJ', 'MCs', 'BaBe']
    s_percents = [p_ktt, p_gy, p_mj, p_mcs, p_babe] # A megadott paraméterek
    t_percents = get_species_percentages(first_df[first_df['T'] == 1], sp_list)
    c_percents = get_species_percentages(first_df[first_df['C'] == 1], sp_list)

    def draw_species_bar(label, percents):
        cols_html = "".join([
            f'<div style="width: {p}%; background-color: {species_colors[sp]}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 10px; min-width: 0; overflow: hidden;">{f"{p:.1f}%" if p > 5 else ""}</div>'
            for p, sp in zip(percents, sp_list) if p > 0
        ])
        st.write(f"**{label}**")
        st.markdown(f'<div style="display: flex; height: 30px; width: 100%; border-radius: 5px; overflow: hidden; border: 1px solid #ddd; margin-bottom: 10px;">{cols_html}</div>', unsafe_allow_html=True)

    draw_species_bar("Valódi összetétel (S)", s_percents)
    draw_species_bar("Transzekt becslés (T)", t_percents)
    draw_species_bar("Mintakör becslés (C)", c_percents)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Magasság eloszlás")
        fig_dist, ax_dist = plt.subplots()
        sns.histplot(df['height'], kde=True, bins=30, color="forestgreen", ax=ax_dist)
        st.pyplot(fig_dist)
        plt.close(fig_dist)
    with col2:
        st.subheader("🧊 3D Nézet")
        fig_3d = plt.figure()
        ax3d = fig_3d.add_subplot(111, projection='3d')
        for sp in sim_params['sp_names']:
            sp_df = df[df['species'] == sp]
            if not sp_df.empty:
                ax3d.scatter(sp_df['X'], sp_df['Y'], sp_df['height'], color=species_colors[sp], s=sp_df['height'], alpha=0.6)
                ax3d.set_zlim(0, 200) # Z-tengely skálája
        st.pyplot(fig_3d)
        plt.close(fig_3d)

    st.subheader("🗺️ Mintavételi térképek")
    m1, m2 = st.columns(2)
    with m1:
        st.write("Transzekt")
        fig_map, ax_map = plt.subplots()
        ax_map.scatter(df['X'], df['Y'], c='lightgrey', s=2, alpha=0.3)
        t_df_plot = df[df['T'] == 1]
        ax_map.scatter(t_df_plot['X'], t_df_plot['Y'], c='red', s=10)
        ax_map.plot([0, width], [0, height], 'r--', lw=1)
        ax_map.set_aspect('equal')
        st.pyplot(fig_map)
    with m2:
        st.write("Mintakörök")
        fig_circ, ax_circ = plt.subplots()
        ax_circ.scatter(df['X'], df['Y'], c='lightgrey', s=2, alpha=0.3)
        c_df_plot = df[df['C'] == 1]
        ax_circ.scatter(c_df_plot['X'], c_df_plot['Y'], c='blue', s=10)
        ax_circ.add_patch(patches.Circle(center_big, r_big, color='navy', fill=False))
        for cs in centers_small: ax_circ.add_patch(patches.Circle(cs, r_small, color='dodgerblue', fill=False))
        ax_circ.set_aspect('equal')
        st.pyplot(fig_circ)

    st.subheader("🦌 Rágottság fafajonként")
    fig_chew, ax_chew = plt.subplots(figsize=(10, 4))
    spec_chew = df.groupby('species')['chewed'].mean() * 100
    ax_chew.bar(spec_chew.index, spec_chew.values, color=[species_colors.get(x) for x in spec_chew.index])
    ax_chew.axhline(in_chewed, color='red', linestyle='--', label='Cél')
    st.pyplot(fig_chew)






