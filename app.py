import streamlit as st
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as patches

# --- 1. ALAPBEÁLLÍTÁSOK ---
st.set_page_config(page_title="Profi Erdő Szimulátor", layout="wide")

width, height = 1500, 1500
total_area = width * height
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

# --- 2. SZIMULÁCIÓS FÜGGVÉNY ---
def run_forest_simulation(params):
    target_intensity = params['intensity']
    # Kicsit több fát generálunk, hogy a Poisson ingadozás után is meglegyen a sűrűség
    expected_n = int(target_intensity * total_area)
    N_gen = np.random.poisson(expected_n)
    
    x = np.random.uniform(0, width, N_gen)
    y = np.random.uniform(0, height, N_gen)
    
    shape_k = 5.0
    heights = np.clip(np.random.gamma(shape=shape_k, scale=params['scale']/(shape_k-1), size=N_gen), 3, 200)
    fajok = np.random.choice(params['sp_names'], size=N_gen, p=params['sp_probs'])
    ragottsag = np.random.uniform(0, 100, size=N_gen) < params['chewed_p']
    
    df = pd.DataFrame({
        "X": x, "Y": y, "height": heights, 
        "species": fajok, "chewed": ragottsag.astype(int)
    })
    
    # Transzekt (T) szűrés: a fa magassága a sáv szélessége
    df['T'] = df.apply(lambda r: 1 if point_line_distance(r['X'], r['Y'], 0, 0, width, height) <= r['height'] else 0, axis=1)
    
    # Mintakör (C) szűrés
    def check_circ(r):
        if r['height'] > 50:
            return 1 if math.dist((r['X'], r['Y']), center_big) <= r_big else 0
        else:
            for cs in centers_small:
                if math.dist((r['X'], r['Y']), cs) <= r_small: return 1
            return 0
    df['C'] = df.apply(check_circ, axis=1)
    
    return df

# --- 3. FELHASZNÁLÓI FELÜLET ---
with st.sidebar:
    st.header("⚙️ Beállítások")
    in_intensity = st.slider("Cél sűrűség (db/m²)", 0.00005, 0.005, 0.0020, step=0.00005, format="%.5f")
    in_scale = st.slider("Magasság scale (módusz)", 5, 50, 15)
    in_chewed = st.slider("Valódi rágottság (%)", 0, 100, 30)
    in_runs = st.slider("Szimulációs futások száma", 2, 50, 5)
    
    st.markdown("---")
    st.subheader("🌿 Fajösszetétel (%)")
    p_ktt = st.number_input("KTT", 0, 100, 20)
    p_gy = st.number_input("Gy", 0, 100, 20)
    p_mj = st.number_input("MJ", 0, 100, 20)
    p_mcs = st.number_input("MCs", 0, 100, 20)
    p_babe = max(0, 100 - (p_ktt + p_gy + p_mj + p_mcs))
    st.info(f"BaBe (maradék): {p_babe}%")

# --- 4. SZIMULÁCIÓ ÉS MEGJELENÍTÉS ---
if st.button("SZIMULÁCIÓ ÉS ELEMZÉS INDÍTÁSA", use_container_width=True):
    probs = np.array([p_ktt, p_gy, p_mj, p_mcs, p_babe], dtype=float)
    if probs.sum() > 0:
        probs /= probs.sum()
    else:
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    sim_params = {
        'intensity': in_intensity, 'scale': in_scale, 'chewed_p': in_chewed,
        'sp_names': ['KTT', 'Gy', 'MJ', 'MCs', 'BaBe'], 'sp_probs': probs
    }
    
    results_list = []
    first_df = None

    my_bar = st.progress(0)
    for i in range(in_runs):
        df = run_forest_simulation(sim_params)
        if i == 0: first_df = df
        
        # S (Valódi adatok)
        s_count = len(df)
        s_dens = s_count / total_area
        s_chew = df['chewed'].mean() * 100
        
        # T (Transzekt becslés)
        t_df = df[df['T'] == 1]
        t_count = len(t_df)
        t_dens = (t_df['height'].apply(lambda h: 1/h).sum() / (math.sqrt(width**2 + height**2))) if t_count > 0 else 0
        t_chew = t_df['chewed'].mean() * 100 if t_count > 0 else 0
        
        # C (Mintakör becslés)
        c_df = df[df['C'] == 1]
        c_count = len(c_df)
        c_large = c_df[c_df['height'] > 50]
        c_small = c_df[c_df['height'] <= 50]
        c_dens = (len(c_large)/area_big_circle + len(c_small)/area_small_circles) if c_count > 0 else 0
        c_chew = c_df['chewed'].mean() * 100 if c_count > 0 else 0
        
        results_list.append([s_count, t_count, c_count, s_dens, t_dens, c_dens, s_chew, t_chew, c_chew])
        my_bar.progress((i + 1) / in_runs)

    res_arr = np.array(results_list)
    means = res_arr.mean(axis=0)
    
    # --- TÁBLÁZAT MEGJELENÍTÉSE ---
    final_summary = pd.DataFrame({
        "Mérés típusa": ["Darabszám (count)", "Sűrűség (density)", "Rágottság (chewed%)"],
        "Valódi (S)": [f"{means[0]:.0f}", f"{means[3]:.5f}", f"{means[6]:.1f}%"],
        "Transzekt (T)": [f"{means[1]:.1f}", f"{means[4]:.5f}", f"{means[7]:.1f}%"],
        "Mintakör (C)": [f"{means[2]:.1f}", f"{means[5]:.5f}", f"{means[8]:.1f}%"]
    })
    
    st.subheader(f"📊 Összesített statisztika ({in_runs} futás átlaga)")
    st.table(final_summary)

    # --- SZÁZALÉKOS SÁV ---
    st.subheader("🌲 Fafaj-összetétel eloszlása")
    percentages = [p_ktt, p_gy, p_mj, p_mcs, p_babe]
    names = ['KTT', 'Gy', 'MJ', 'MCs', 'BaBe']
    
    html_bar = '<div style="display: flex; height: 45px; width: 100%; border-radius: 10px; overflow: hidden; border: 2px solid #333; margin-bottom: 25px;">'
    for p, name in zip(percentages, names):
        if p > 0:
            html_bar += f'''<div style="width: {p}%; background-color: {species_colors[name]}; 
                            display: flex; align-items: center; justify-content: center; 
                            color: white; font-weight: bold; font-size: 14px; border-right: 1px solid rgba(255,255,255,0.3);">
                            {name}: {p}%</div>'''
    html_bar += '</div>'
    st.markdown(html_bar, unsafe_allow_html=True)

    # --- TÉRKÉPEK ---
    df = first_df
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🗺️ Transzekt (T)")
        fig_t, ax_t = plt.subplots(figsize=(10,10))
        # Sötétített háttér pontok
        ax_t.scatter(df['X'], df['Y'], c='dimgray', s=7, alpha=0.5, label='Nem mért')
        t_pts = df[df['T'] == 1]
        for sp in names:
            curr = t_pts[t_pts['species'] == sp]
            if not curr.empty:
                ax_t.scatter(curr['X'], curr['Y'], c=species_colors[sp], s=35, label=sp, edgecolors='black', lw=0.5)
        ax_t.plot([0, width], [0, height], color='red', linestyle='--', lw=2, label='Tengely')
        ax_t.set_xlim(0, width); ax_t.set_ylim(0, height)
        ax_t.set_aspect('equal')
        ax_t.legend(loc='upper right')
        st.pyplot(fig_t)

    with col2:
        st.write("### 🎯 Mintakörök (C)")
        fig_c, ax_c = plt.subplots(figsize=(10,10))
        # Sötétített háttér pontok
        ax_c.scatter(df['X'], df['Y'], c='dimgray', s=7, alpha=0.5)
        c_pts = df[df['C'] == 1]
        for sp in names:
            curr = c_pts[c_pts['species'] == sp]
            if not curr.empty:
                ax_c.scatter(curr['X'], curr['Y'], c=species_colors[sp], s=35, edgecolors='black', lw=0.5)
        
        # Körök rajzolása
        ax_c.add_patch(patches.Circle(center_big, r_big, color='navy', fill=False, lw=3, label='Nagy kör'))
        for cs in centers_small:
            ax_c.add_patch(patches.Circle(cs, r_small, color='dodgerblue', fill=False, lw=2, ls='--'))
        
        ax_c.set_xlim(0, width); ax_c.set_ylim(0, height)
        ax_c.set_aspect('equal')
        st.pyplot(fig_c)

    # --- RÁGOTTSÁG ---
    st.subheader("🦌 Rágottság fafajonként (Első futás)")
    fig_b, ax_b = plt.subplots(figsize=(12, 5))
    # Csoportosítás és átlag számítás
    avg_chew = df.groupby('species')['chewed'].mean() * 100
    # Biztosítjuk, hogy minden faj szerepeljen az x tengelyen
    plot_data = [avg_chew.get(s, 0) for s in names]
    
    bars = ax_b.bar(names, plot_data, color=[species_colors[s] for s in names], edgecolor='black')
    ax_b.axhline(in_chewed, color='red', ls='--', lw=2, label=f'Cél ({in_chewed}%)')
    
    # Értékek kiírása az oszlopok fölé
    for bar in bars:
        height_val = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., height_val + 1, f'{height_val:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax_b.set_ylim(0, 115)
    ax_b.set_ylabel("Rágottság mértéke (%)")
    ax_b.legend()
    st.pyplot(fig_b)
