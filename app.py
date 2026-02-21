import streamlit as st
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. ALAPBEÁLLÍTÁSOK ---
st.set_page_config(page_title="Profi Erdő Szimulátor", layout="centered")

width, height = 1500, 1500
max_height = 200
min_height = 3
R_core = 5  # Minimum távolság két fa között (cm/pixel)
center_big = (width/2, height/2)
r_big = 564
r_small = 126
centers_small = [(width/4, height/4), (3*width/4, height/4), 
                 (width/4, 3*height/4), (3*width/4, 3*height/4)]

def point_line_distance(x, y, x1, y1, x2, y2):
    num = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1))
    den = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return num / den

# --- 2. KORREKCIÓS MOTOR (Matérn szűrés kompenzálása) ---
def get_retention_ratio(intensity, R_core):
    # Gyorsított becslés a veszteségre
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
    # Intenzitás korrekció, hogy a szűrés után is annyi fa maradjon, amennyit kértünk
    retention = get_retention_ratio(target_intensity, R_core)
    corrected_intensity = target_intensity / max(retention, 0.1)
    
    expected_n = int(corrected_intensity * width * height)
    N_gen = np.random.poisson(expected_n)
    
    # Gravitációs pontok generálása
    grav_centers = np.random.uniform(0, width, (params['n_grav'], 2))
    
    # Pontok generálása csoportosulással
    N_oversample = N_gen * 5
    x_tmp = np.random.uniform(0, width, N_oversample)
    y_tmp = np.random.uniform(0, height, N_oversample)
    
    # Csoportosulás számítása
    dist_all = np.array([np.sqrt((x_tmp - cx)**2 + (y_tmp - cy)**2) for cx, cy in grav_centers])
    min_dists = dist_all.min(axis=0)
    weights = np.exp(-min_dists**2 / (2 * params['sigma']**2))
    weights /= weights.max()
    
    # Sűrűsödési erő alkalmazása
    keep_mask = np.random.uniform(0, 1, N_oversample) < weights
    accepted = np.column_stack((x_tmp, y_tmp))[keep_mask]
    
    if len(accepted) > N_gen:
        accepted = accepted[np.random.choice(len(accepted), N_gen, replace=False)]
    
    # Matérn szűrés (ne legyenek túl közel)
    final_keep = np.ones(len(accepted), dtype=bool)
    R_sq = R_core**2
    for i in range(len(accepted)):
        if not final_keep[i]: continue
        d_sq = np.sum((accepted[i] - accepted)**2, axis=1)
        final_keep[(d_sq < R_sq) & (d_sq > 0)] = False
    
    final_coords = accepted[final_keep]
    N_final = len(final_coords)
    
    # Magasság (Gamma eloszlás + Gravitációs hatás)
    dist_final = np.array([np.sqrt((final_coords[:,0] - cx)**2 + (final_coords[:,1] - cy)**2) for cx, cy in grav_centers]).min(axis=0)
    shape_k = 2.0
    base_h = min_height + np.random.gamma(shape=shape_k, scale=params['scale']/shape_k, size=N_final)
    gauss_eff = np.exp(-0.5 * (dist_final / params['sigma_h'])**2)
    heights = np.clip(base_h * (1 + params['grav_str'] * gauss_eff), min_height, max_height)
    
    # Fajok és rágottság
    fajok = np.random.choice(params['sp_names'], size=N_final, p=params['sp_probs'])
    ragottsag = np.random.uniform(0, 100, size=N_final) < params['chewed_p']
    
    # Mintavétel ellenőrzése
    results = []
    for i in range(N_final):
        x, y, h = final_coords[i,0], final_coords[i,1], heights[i]
        
        # Transzekt
        in_t = 1 if point_line_distance(x, y, 0, 0, width, height) <= h else 0
        
        # Mintakör
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
st.title("🌲 Profi Erdő Szimulátor v2.0")

with st.sidebar:
    st.header("⚙️ Beállítások")
    # ... (a többi csúszka marad)

    st.subheader("🌿 Fajösszetétel eloszlása")
    
    # 5 darab csúszka, amik összege 100 kell legyen
    p_ktt = st.slider("KTT (%)", 0, 100, 20)
    # Kiszámoljuk, mennyi maradt a többihez
    rem1 = 100 - p_ktt
    
    p_gy = st.slider("Gy (%)", 0, rem1, min(20, rem1))
    rem2 = rem1 - p_gy
    
    p_mj = st.slider("MJ (%)", 0, rem2, min(20, rem2))
    rem3 = rem2 - p_mj
    
    p_mcs = st.slider("MCs (%)", 0, rem3, min(20, rem3))
    rem4 = rem3 - p_mcs
    
    p_babe = rem4  # Ami maradt, az automatikusan a BaBe
    
    st.info(f"BaBe (maradék): {p_babe}%")
    
    total_p = p_ktt + p_gy + p_mj + p_mcs + p_babe
    st.write(f"**Összesen: {total_p}%**")
if st.button("SZIMULÁCIÓ FUTTATÁSA", use_container_width=True) and total_p == 100:
    sim_params = {
        'intensity': in_intensity, 'scale': in_scale, 'grav_str': in_grav_str,
        'chewed_p': in_chewed, 'n_grav': 3, 'sigma': 400, 'sigma_h': 50.0,
        'sp_names': ['KTT', 'Gy', 'MJ', 'MCs', 'BaBe'],
        'sp_probs': [p_ktt/100, p_gy/100, p_mj/100, p_mcs/100, p_babe/100]
    }
    
    with st.spinner("Erdő generálása..."):
        df = run_forest_simulation(sim_params)
    
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Összes fa", len(df))
            st.metric("Transzekt találat", df['T'].sum())
        with col2:
            st.metric("Rágott egyedek", df['chewed'].sum())
            st.metric("Kör találat", df['C'].sum())

        # Vizualizáció
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(data=df, x="X", y="Y", hue="species", size="height", 
                        style="chewed", markers={0: 'o', 1: 'X'}, alpha=0.6, ax=ax)
        
        # Mintavételi területek jelölése
        ax.plot([0, 1500], [0, 1500], 'r--', alpha=0.3, label="Transzekt")
        ax.add_patch(patches.Circle(center_big, r_big, color='blue', fill=False, linestyle='--', label="Nagy kör"))
        for cs in centers_small:
            ax.add_patch(patches.Circle(cs, r_small, color='green', fill=False, label="Kis körök"))
            
        ax.set_xlim(0, 1500)
        ax.set_ylim(0, 1500)
        ax.set_aspect('equal')
        st.pyplot(fig)
    else:
        st.error("Nem sikerült fákat generálni. Próbáld nagyobb sűrűséggel!")
