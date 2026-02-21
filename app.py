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

# --- 3. FELHASZNÁLÓI FELÜLET ---
with st.sidebar:
    st.header("⚙️ Beállítások")
    in_intensity = st.slider("Cél sűrűség (db/m²)", 0.0005, 0.0100, 0.0020, step=0.0005, format="%.4f")
    in_scale = st.slider("Magasság scale (módusz)", 5, 50, 15)
    in_grav_str = st.slider("Sűrűsödési erő", 0, 10, 3)
    in_chewed = st.slider("Valódi rágottság (%)", 0, 100, 30)
    
    st.markdown("---")
    st.subheader("🌿 Fajösszetétel (Interaktív)")

    # Alapértékek beállítása, ha még nincsenek
    if 'KTT' not in st.session_state: st.session_state['KTT'] = 20
    if 'Gy' not in st.session_state: st.session_state['Gy'] = 20
    if 'MJ' not in st.session_state: st.session_state['MJ'] = 20
    if 'MCs' not in st.session_state: st.session_state['MCs'] = 20

    def sync_sliders(changed_key):
        """Ez a függvény gondoskodik róla, hogy ne lépjük túl a 100%-ot."""
        current_total = st.session_state['KTT'] + st.session_state['Gy'] + st.session_state['MJ'] + st.session_state['MCs']
        
        if current_total > 100:
            # Mennyit kell levonnunk a többiekből?
            excess = current_total - 100
            others = [k for k in ['KTT', 'Gy', 'MJ', 'MCs'] if k != changed_key]
            
            # Levonjuk a felesleget a többi csúszkából (sorrendben, amíg el nem fogy az excess)
            for k in others:
                if st.session_state[k] >= excess:
                    st.session_state[k] -= excess
                    excess = 0
                    break
                else:
                    excess -= st.session_state[k]
                    st.session_state[k] = 0

    # A csúszkák, amik a session_state-et használják
    p_ktt = st.slider("KTT (%)", 0, 100, key='KTT', on_change=sync_sliders, args=('KTT',))
    p_gy = st.slider("Gy (%)", 0, 100, key='Gy', on_change=sync_sliders, args=('Gy',))
    p_mj = st.slider("MJ (%)", 0, 100, key='MJ', on_change=sync_sliders, args=('MJ',))
    p_mcs = st.slider("MCs (%)", 0, 100, key='MCs', on_change=sync_sliders, args=('MCs',))

    # A maradék BaBe
    p_babe = max(0, 100 - (p_ktt + p_gy + p_mj + p_mcs))
    
    st.info(f"BaBe: {p_babe}%")
    
    st.info(f"BaBe (maradék): {p_babe}%")
    st.caption(f"Összesen: {p_ktt + p_gy + p_mj + p_mcs + p_babe}%")
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
       # --- JAVÍTOTT STATISZTIKA ---
        t_df = df[df['T'] == 1]
        c_df = df[df['C'] == 1]

        # Mintakör sűrűség becslése (db/m2)
        # Külön választjuk a kicsiket és nagyokat a mintában
        c_large = c_df[c_df['height'] > 50]
        c_small = c_df[c_df['height'] <= 50]
        
        # Sűrűség = (Nagyok száma / Nagy kör területe) + (Kicsik száma / Kis körök összterülete)
        if area_big_circle > 0 and area_small_circles > 0:
            c_density_estimate = (len(c_large) / area_big_circle) + (len(c_small) / area_small_circles)
        else:
            c_density_estimate = 0

        stats_data = {
            "Paraméter": ["Egyedszám", "Sűrűség (db/m²)", "Scale (H)", "Rágottság"],
            "Valódi (S)": [
                len(df), 
                f"{len(df)/(width*height):.4f}", 
                get_weighted_height_mode(df), 
                f"{df['chewed'].mean()*100:.1f}%"
            ],
            "Transzekt (T)": [
                len(t_df), 
                f"{(t_df['height'].apply(lambda h: 1/h).sum()/width) if len(t_df)>0 else 0:.4f}", 
                get_weighted_height_mode(t_df, True), 
                f"{t_df['chewed'].mean()*100 if len(t_df)>0 else 0:.1f}%"
            ],
            "Mintakör (C)": [
                len(c_df), 
                f"{c_density_estimate:.4f}", # <--- Itt már nem N/A van!
                get_weighted_height_mode(c_df), 
                f"{c_df['chewed'].mean()*100 if len(c_df)>0 else 0:.1f}%"
            ]
        }
        st.subheader("📊 Becslési eredmények")
        st.table(pd.DataFrame(stats_data))
        st.markdown("---")
        st.subheader("🌲 A szimulált erdő fafaj-összetétele")
        
        # Egy látványos, színes sávdiagram HTML/CSS segítségével
        st.markdown(
            f"""
            <div style="display: flex; height: 35px; width: 100%; border-radius: 8px; overflow: hidden; border: 2px solid #ddd; margin-bottom: 20px;">
                <div style="width: {p_ktt}%; background-color: {species_colors['KTT']}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{p_ktt if p_ktt > 5 else ''}%</div>
                <div style="width: {p_gy}%; background-color: {species_colors['Gy']}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{p_gy if p_gy > 5 else ''}%</div>
                <div style="width: {p_mj}%; background-color: {species_colors['MJ']}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{p_mj if p_mj > 5 else ''}%</div>
                <div style="width: {p_mcs}%; background-color: {species_colors['MCs']}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{p_mcs if p_mcs > 5 else ''}%</div>
                <div style="width: {p_babe}%; background-color: {species_colors['BaBe']}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{p_babe if p_babe > 5 else ''}%</div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px; font-size: 14px;">
                <span style="color: {species_colors['KTT']};">■ KTT</span>
                <span style="color: {species_colors['Gy']};">■ Gy</span>
                <span style="color: {species_colors['MJ']};">■ MJ</span>
                <span style="color: {species_colors['MCs']};">■ MCs</span>
                <span style="color: {species_colors['BaBe']};">■ BaBe</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("---")
      
      # --- MAGASSÁG ELOSZLÁSI GÖRBE ---
        st.subheader("📊 Magasság eloszlás az erdőben")
        
        fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
        
        # Hisztogram és simított görbe (KDE)
        sns.histplot(df['height'], kde=True, bins=30, color="forestgreen", ax=ax_dist, stat="density")
        
        # Átlag és Módusz jelölése
        mean_h = df['height'].mean()
        mode_h = get_weighted_height_mode(df)
        
        ax_dist.axvline(mean_h, color='red', linestyle='--', label=f'Átlag: {mean_h:.1f} m')
        ax_dist.axvline(mode_h, color='blue', linestyle=':', label=f'Módusz: {mode_h:.1f} m')
        
        ax_dist.set_xlabel("Magasság (m)")
        ax_dist.set_ylabel("Gyakoriság")
        ax_dist.legend()
        
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        
        st.markdown("---")

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
      # --- 4. TRANSZEKT FELÜLNÉZETI TÉRKÉP ---
        st.subheader("🗺️ Transzekt mintavétel felülnézetből")
        fig_map, ax_map = plt.subplots(figsize=(10, 10))
        
        # Az összes fa halványan a háttérben
        ax_map.scatter(df['X'], df['Y'], c='lightgrey', s=5, alpha=0.3, label='Erdő egyedei')
        
        # A transzektbe eső fák kiemelve fajonkénti színnel
        t_df = df[df['T'] == 1]
        if not t_df.empty:
            for sp in sim_params['sp_names']:
                sp_t = t_df[t_df['species'] == sp]
                if not sp_t.empty:
                    ax_map.scatter(sp_t['X'], sp_t['Y'], color=species_colors[sp], s=20, label=f'{sp} (mintában)')

        # Az átló (transzekt középvonala)
        ax_map.plot([0, width], [0, height], color='red', linestyle='--', linewidth=1, label='Transzekt tengely')
        
        # Transzekt határának szemléltetése (elvi sáv)
        # Mivel a szélesség fánként változik (height), egy átlagos sávot rajzolunk a szemléltetéshez
        avg_h = df['height'].mean()
        ax_map.fill_between([0, width], [0 - avg_h, height - avg_h], [0 + avg_h, height + avg_h], 
                            color='red', alpha=0.1, label='Átlagos mintavételi sáv')

        ax_map.set_xlim(0, width)
        ax_map.set_ylim(0, height)
        ax_map.set_aspect('equal')
        ax_map.legend(loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig_map)
        plt.close(fig_map)
        
        st.markdown("---")

        # --- 5. RÁGOTTSÁGI STATISZTIKA (FAJONKÉNT) ---
        st.subheader("🦌 Rágottság mértéke fafajonként(így szar majd más lesz)")
        fig_chew, ax_chew = plt.subplots(figsize=(10, 5))
        
        # Kiszámoljuk a fajonkénti rágottsági arányt
        species_chewed = df.groupby('species')['chewed'].mean() * 100
        
        # Biztosítjuk, hogy minden faj szerepeljen a grafikonon, akkor is ha 0%
        full_species_list = sim_params['sp_names']
        chew_values = [species_chewed.get(sp, 0) for sp in full_species_list]
        colors = [species_colors[sp] for sp in full_species_list]
        
        bars = ax_chew.bar(full_species_list, chew_values, color=colors, edgecolor='black', alpha=0.8)
        
        # Értékek ráírása az oszlopokra
        for bar in bars:
            yval = bar.get_height()
            ax_chew.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax_chew.axhline(in_chewed, color='red', linestyle='--', label=f'Beállított cél ({in_chewed}%)')
        ax_chew.set_ylabel("Rágott egyedek aránya (%)")
        ax_chew.set_ylim(0, 110)
        ax_chew.legend()
        st.pyplot(fig_chew)
        plt.close(fig_chew)

        st.markdown("---")
      # --- 5. MINTAKÖRÖS FELÜLNÉZETI TÉRKÉP ---
        st.subheader("🎯 Mintakörös mintavétel felülnézetből")
        fig_circ, ax_circ = plt.subplots(figsize=(10, 10))
        
        # Az összes fa halványan a háttérben
        ax_circ.scatter(df['X'], df['Y'], c='lightgray', s=5, alpha=0.3, label='Erdő egyedei')
        
        # A mintakörökbe eső fák kiemelve (C == 1)
        c_df = df[df['C'] == 1]
        if not c_df.empty:
            for sp in sim_params['sp_names']:
                sp_c = c_df[c_df['species'] == sp]
                if not sp_c.empty:
                    ax_circ.scatter(sp_c['X'], sp_c['Y'], color=species_colors[sp], s=30, 
                                    label=f'{sp} (mintában)', edgecolors='white', linewidth=0.5)

        # A geometria berajzolása (Körök)
        # 1. Nagy mintakör (h > 50 egyedeknek)
        circle_big_patch = patches.Circle(center_big, r_big, color='navy', fill=False, 
                                          linestyle='--', linewidth=2, alpha=0.5, label='Nagy mintakör (h>50)')
        ax_circ.add_patch(circle_big_patch)
        
        # 2. Négy kis mintakör (h <= 50 egyedeknek)
        for i, cs in enumerate(centers_small):
            lbl = 'Kis mintakörök (h<=50)' if i == 0 else ""
            circle_small_patch = patches.Circle(cs, r_small, color='dodgerblue', fill=False, 
                                                linestyle=':', linewidth=2, alpha=0.6, label=lbl)
            ax_circ.add_patch(circle_small_patch)

        # Gravitációs pontok (Csillagok), ha van sűrűsödési erő
        if in_grav_str > 0:
            # A szimulációs függvényben használt grav_centers-t itt újra elérhetjük
            # (Feltételezve, hogy a függvény visszadja vagy fix helyen vannak)
            # Itt most jelképesen a gravitációs központok megjelenítése:
            ax_circ.text(10, 1450, f"Sűrűsödési erő: {in_grav_str}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        ax_circ.set_xlim(0, width)
        ax_circ.set_ylim(0, height)
        ax_circ.set_aspect('equal')
        ax_circ.set_xlabel("X koordináta (m)")
        ax_circ.set_ylabel("Y koordináta (m)")
        ax_circ.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Jelmagyarázat")
        
        st.pyplot(fig_circ)
        plt.close(fig_circ)
        
        st.markdown("---")










