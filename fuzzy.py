import streamlit as st
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Kelayakan Hunian Pasca Bencana",
    page_icon="🏚️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-radius: 14px; padding: 1.8rem 1.5rem;
        margin-bottom: 1.5rem; color: white; text-align: center;
    }
    .hero h1 { font-size: 1.5rem; font-weight: 700; margin: 0.4rem 0; }
    .hero p  { font-size: 0.85rem; opacity: 0.75; margin: 0; }

    .card {
        background: white; border-radius: 12px;
        padding: 1.25rem; border: 1px solid #e9ecef;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.09em; color: #6c757d; margin-bottom: 1rem;
    }

    .result-layak     { background: #d4edda; border: 2px solid #28a745; border-radius: 12px; padding: 1.25rem; text-align: center; }
    .result-bersyarat { background: #fff3cd; border: 2px solid #ffc107; border-radius: 12px; padding: 1.25rem; text-align: center; }
    .result-tidak     { background: #f8d7da; border: 2px solid #dc3545; border-radius: 12px; padding: 1.25rem; text-align: center; }
    .result-score { font-size: 3rem; font-weight: 700; margin: 0.3rem 0; line-height: 1; }
    .result-label { font-size: 1.1rem; font-weight: 600; }

    .badge { display: inline-block; padding: 0.18rem 0.55rem; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
    .badge-ringan { background: #d4edda; color: #155724; }
    .badge-sedang { background: #fff3cd; color: #856404; }
    .badge-berat  { background: #f8d7da; color: #721c24; }

    .rule-item      { background: #f8f9fa; border-left: 3px solid #dee2e6; border-radius: 0 6px 6px 0; padding: 0.45rem 0.7rem; margin-bottom: 0.35rem; font-size: 0.82rem; }
    .rule-layak     { border-left-color: #28a745; background: #f0fff4; color: #155724; }
    .rule-bersyarat { border-left-color: #ffc107; background: #fffdf0; color: #856404; }
    .rule-tidak     { border-left-color: #dc3545; background: #fff5f5; color: #721c24; }

    .mem-box { background: #f8f9fa; border-radius: 8px; padding: 0.6rem 0.5rem; text-align: center; border: 1px solid #e9ecef; margin-bottom: 0.4rem; }
    .mem-val { font-size: 1.25rem; font-weight: 700; }
    .mem-lbl { font-size: 0.68rem; color: #6c757d; }

    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #1a5276);
        color: white; border: none; border-radius: 8px;
        padding: 0.55rem 1.5rem; font-weight: 600;
        font-size: 0.95rem; width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
    
    .info-box {
        background: #e8f4fd; border-left: 3px solid #3498db;
        border-radius: 0 8px 8px 0; padding: 0.6rem 0.8rem;
        font-size: 0.82rem; color: #1a5276; margin-bottom: 0.75rem;
    }
    .footer { text-align: center; color: #adb5bd; font-size: 0.78rem; padding: 1.25rem 0; border-top: 1px solid #e9ecef; margin-top: 1.5rem; }
    div[data-testid="stExpander"] { border: 1px solid #e9ecef; border-radius: 10px; }
    
    .warning-box {
        background: #fff3cd; border-left: 3px solid #ffc107;
        border-radius: 0 8px 8px 0; padding: 0.5rem 0.8rem;
        font-size: 0.8rem; color: #856404; margin: 0.5rem 0;
    }
    
    .confidence-bar {
        background: #e9ecef; border-radius: 10px; height: 6px; margin: 8px 0;
    }
    .history-item {
        background: #f8f9fa; border-radius: 8px; padding: 0.5rem;
        margin-bottom: 0.5rem; font-size: 0.8rem;
        border-left: 3px solid #dee2e6;
    }
    .history-layak { border-left-color: #28a745; }
    .history-bersyarat { border-left-color: #ffc107; }
    .history-tidak { border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# ── Universe of discourse
u_ker = np.arange(0, 101, 1)
u_gen = np.arange(0, 201, 1)
u_aks = np.arange(0, 15.1, 0.1)
u_out = np.arange(0, 101, 1)

# ── Membership functions
mf_ker = {
    'ringan': fuzz.trapmf(u_ker, [0, 0, 20, 40]),
    'sedang': fuzz.trimf(u_ker, [30, 50, 70]),
    'berat': fuzz.trapmf(u_ker, [60, 80, 100, 100]),
}
mf_gen = {
    'rendah': fuzz.trapmf(u_gen, [0, 0, 20, 40]),
    'sedang': fuzz.trimf(u_gen, [30, 60, 100]),
    'tinggi': fuzz.trapmf(u_gen, [80, 120, 200, 200]),
}
mf_aks = {
    'dekat': fuzz.trapmf(u_aks, [0, 0, 2, 4]),
    'sedang': fuzz.trimf(u_aks, [3, 5.5, 8]),
    'jauh': fuzz.trapmf(u_aks, [7, 10, 15, 15]),
}
mf_out = {
    'tidak': fuzz.trapmf(u_out, [0, 0, 20, 40]),
    'bersyarat': fuzz.trimf(u_out, [35, 52.5, 70]),
    'layak': fuzz.trapmf(u_out, [65, 80, 100, 100]),
}

RULES = [
    (1, 'ringan', 'rendah', 'dekat', 'layak', 'R1: Ringan ∧ Rendah ∧ Dekat → Layak'),
    (2, 'ringan', 'rendah', 'sedang', 'bersyarat', 'R2: Ringan ∧ Rendah ∧ Sedang → Layak Bersyarat'),
    (3, 'ringan', 'sedang', 'dekat', 'bersyarat', 'R3: Ringan ∧ Sedang ∧ Dekat → Layak Bersyarat'),
    (4, 'ringan', 'sedang', 'sedang', 'bersyarat', 'R4: Ringan ∧ Sedang ∧ Sedang → Layak Bersyarat'),
    (5, 'ringan', 'tinggi', None, 'tidak', 'R5: Ringan ∧ Tinggi ∧ [any] → Tidak Layak'),
    (6, 'ringan', None, 'jauh', 'tidak', 'R6: Ringan ∧ [any] ∧ Jauh → Tidak Layak'),
    (7, 'sedang', 'rendah', 'dekat', 'bersyarat', 'R7: Sedang ∧ Rendah ∧ Dekat → Layak Bersyarat'),
    (8, 'sedang', 'rendah', 'sedang', 'bersyarat', 'R8: Sedang ∧ Rendah ∧ Sedang → Layak Bersyarat'),
    (9, 'sedang', 'sedang', 'dekat', 'tidak', 'R9: Sedang ∧ Sedang ∧ Dekat → Tidak Layak'),
    (10, 'sedang', 'sedang', 'sedang', 'tidak', 'R10: Sedang ∧ Sedang ∧ Sedang → Tidak Layak'),
    (11, 'sedang', 'tinggi', None, 'tidak', 'R11: Sedang ∧ Tinggi ∧ [any] → Tidak Layak'),
    (12, 'sedang', None, 'jauh', 'tidak', 'R12: Sedang ∧ [any] ∧ Jauh → Tidak Layak'),
    (13, 'berat', None, None, 'tidak', 'R13: Berat ∧ [any] ∧ [any] → Tidak Layak'),
    (14, None, 'tinggi', None, 'tidak', 'R14: [any] ∧ Tinggi ∧ [any] → Tidak Layak'),
    (15, None, None, 'jauh', 'tidak', 'R15: [any] ∧ [any] ∧ Jauh → Tidak Layak'),
    (16, 'ringan', 'rendah', 'jauh', 'tidak', 'R16: Ringan ∧ Rendah ∧ Jauh → Tidak Layak'),
]

def get_membership(universe, mf_dict, val):
    return {k: float(fuzz.interp_membership(universe, v, val)) for k, v in mf_dict.items()}

def infer(ker_v, gen_v, aks_v):
    km = get_membership(u_ker, mf_ker, ker_v)
    gm = get_membership(u_gen, mf_gen, gen_v)
    am = get_membership(u_aks, mf_aks, aks_v)

    agg = {k: np.zeros_like(u_out, float) for k in ['tidak', 'bersyarat', 'layak']}
    active = []

    for rno, r_k, r_g, r_a, r_out, label in RULES:
        s = []
        if r_k:
            s.append(km[r_k])
        if r_g:
            s.append(gm[r_g])
        if r_a:
            s.append(am[r_a])
        strength = min(s) if s else 0.0
        if strength > 0.001:
            active.append((rno, label, strength, r_out))
            agg[r_out] = np.fmax(agg[r_out], np.fmin(strength, mf_out[r_out]))

    aggregated = np.fmax(agg['tidak'], np.fmax(agg['bersyarat'], agg['layak']))
    try:
        score = fuzz.defuzz(u_out, aggregated, 'centroid')
    except:
        score = 0.0

    return score, sorted(active, key=lambda x: x[2], reverse=True), \
        (agg['tidak'], agg['bersyarat'], agg['layak'], aggregated), (km, gm, am)

def plot_mf(ker_v, gen_v, aks_v):
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.5))
    fig.patch.set_facecolor('#ffffff')

    configs = [
        (axes[0, 0], u_ker,
         [('Ringan', '#28a745', mf_ker['ringan']), ('Sedang', '#ffc107', mf_ker['sedang']),
          ('Berat', '#dc3545', mf_ker['berat'])],
         ker_v, 'Kerusakan Struktur', 'Nilai (0-100)'),
        (axes[0, 1], u_gen,
         [('Rendah', '#28a745', mf_gen['rendah']), ('Sedang', '#ffc107', mf_gen['sedang']),
          ('Tinggi', '#dc3545', mf_gen['tinggi'])],
         gen_v, 'Tinggi Genangan (cm)', 'Nilai (0-200 cm)'),
        (axes[1, 0], u_aks,
         [('Dekat', '#28a745', mf_aks['dekat']), ('Sedang', '#ffc107', mf_aks['sedang']),
          ('Jauh', '#dc3545', mf_aks['jauh'])],
         aks_v, 'Akses Fasilitas (km)', 'Nilai (0-15 km)'),
        (axes[1, 1], u_out,
         [('Tidak Layak', '#dc3545', mf_out['tidak']), ('Layak Bersyarat', '#ffc107', mf_out['bersyarat']),
          ('Layak', '#28a745', mf_out['layak'])],
         None, 'Output: Tingkat Kelayakan', 'Skor (0-100)'),
    ]

    for ax, univ, sets, val, title, xlabel in configs:
        ax.set_facecolor('#fafafa')
        for lbl, col, mf in sets:
            ax.plot(univ, mf, color=col, linewidth=1.5, label=lbl)
            ax.fill_between(univ, mf, alpha=0.1, color=col)
        if val is not None:
            ax.axvline(x=val, color='#0f3460', linewidth=1.5, linestyle='--', alpha=0.85, label=f'Input: {val}')
        ax.set_title(title, fontsize=9, fontweight='600', pad=6)
        ax.set_xlabel(xlabel, fontsize=7.5, color='#6c757d')
        ax.set_ylabel('Derajat Keanggotaan', fontsize=7.5, color='#6c757d')
        ax.set_ylim(-0.05, 1.15)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=1.2)
    return fig

def plot_defuzz(score, agg_data):
    agg_tidak, agg_bersyarat, agg_layak, aggregated = agg_data
    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')
    ax.fill_between(u_out, agg_tidak, alpha=0.5, color='#dc3545', label='Tidak Layak')
    ax.fill_between(u_out, agg_bersyarat, alpha=0.5, color='#ffc107', label='Layak Bersyarat')
    ax.fill_between(u_out, agg_layak, alpha=0.5, color='#28a745', label='Layak')
    ax.plot(u_out, aggregated, 'k-', linewidth=1.2, alpha=0.5, label='Agregasi')
    ax.axvline(x=score, color='#0f3460', linewidth=2, linestyle='--', label=f'COG = {score:.1f}')
    ax.set_title('Agregasi Output & Defuzzifikasi (Centroid)', fontsize=9, fontweight='600')
    ax.set_xlabel('Skor Kelayakan (0-100)', fontsize=8)
    ax.set_ylabel('Derajat', fontsize=8)
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

def validate_input(ker_v, gen_v, aks_v):
    warnings_list = []
    if ker_v >= 80 and gen_v >= 100:
        warnings_list.append("⚠️ Kerusakan berat DAN genangan tinggi → risiko sangat tinggi!")
    if gen_v >= 150:
        warnings_list.append("⚠️ Genangan sangat tinggi (>150 cm) → EVAKUASI SEGERA!")
    if ker_v >= 90:
        warnings_list.append("⚠️ Kerusakan ekstrem (>90) → bangunan berisiko rubuh!")
    if aks_v >= 12:
        warnings_list.append("⚠️ Akses sangat jauh (>12 km) → bantuan darurat sulit dijangkau!")
    return warnings_list

def get_detailed_rekomendasi(score, ker_v, gen_v, aks_v):
    if score >= 65:
        return """
📋 **Rekomendasi Tindakan (Layak Huni):**
- ✅ Rumah aman untuk ditempati
- 🔍 Lakukan pemeriksaan berkala setiap minggu
- 🧹 Bersihkan lumpur dan puing-puing sisa bencana
- 💧 Periksa kualitas air sumur sebelum digunakan
- 📞 Laporkan kerusakan ringan ke RT/RW untuk didata
- 🧴 Lakukan fumigasi untuk mencegah penyakit pasca banjir
"""
    elif score >= 35:
        tindakan = """
📋 **Rekomendasi Tindakan (Layak Bersyarat):**
- ⚠️ Rumah dapat ditempati DENGAN SYARAT
- 🔧 Perbaiki kerusakan struktural minor terlebih dahulu
- 🧯 Pastikan tidak ada kebocoran gas atau korsleting listrik
- 🎒 Siapkan tas darurat di dekat pintu keluar
- 🚪 Jangan tidur di ruangan yang masih basah atau retak
- 📻 Pantau informasi cuaca dan peringatan dini dari BMKG
"""
        if ker_v > 50:
            tindakan += "\n- 🏗️ Prioritas perbaiki dinding yang retak sebelum ditempati"
        if gen_v > 50:
            tindakan += "\n- 💨 Pastikan sirkulasi udara baik untuk mengeringkan bangunan"
        return tindakan
    else:
        tindakan = """
📋 **TINDAKAN DARURAT (Tidak Layak Huni):**
- 🚨 SEGERA EVAKUASI! Jangan tempati bangunan ini
- 📞 Hubungi BPBD: 117 atau call center darurat 112
- 🎒 Bawa dokumen penting, obat-obatan, dan perlengkapan dasar
- ⚡ Matikan aliran listrik dan gas sebelum meninggalkan rumah
- 🏕️ Cari tempat pengungsian terdekat yang sudah ditentukan
- 📢 Ikuti instruksi dari petugas SAR dan BPBD
"""
        if ker_v > 70:
            tindakan += "\n- 🧱 Bangunan berisiko rubuh, jangan ambil barang berharga"
        if gen_v > 100:
            tindakan += "\n- 🌊 Arus air deras, jangan nekat berjalan di genangan"
        return tindakan

# INITIALIZE SESSION STATE
if 'riwayat' not in st.session_state:
    st.session_state.riwayat = []
if 'hasil_dihitung' not in st.session_state:
    st.session_state.hasil_dihitung = False
if 'score' not in st.session_state:
    st.session_state.score = 50
if 'active_rules' not in st.session_state:
    st.session_state.active_rules = []
if 'agg_data' not in st.session_state:
    st.session_state.agg_data = None
if 'memberships' not in st.session_state:
    st.session_state.memberships = None

# HEADER
st.markdown("""
<div class="hero">
    <div style="font-size:2rem">🏚️</div>
    <h1>Sistem Penilaian Kelayakan Tempat Tinggal Pasca Bencana</h1>
    <p>Sistem Inferensi Fuzzy Mamdani · Defuzzifikasi Centroid (COG)<br>
    Menilai kelayakan hunian setelah banjir, gempa, atau tanah longsor</p>
</div>
""", unsafe_allow_html=True)

# LAYOUT
col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.markdown('<div class="card"><div class="card-title">📊 Masukkan Data Kondisi Rumah</div>', unsafe_allow_html=True)

    ker_v = st.slider(
        "🏗️ Kerusakan Struktur Bangunan",
        min_value=0, max_value=100, value=40, step=1,
        help="0 = tidak ada kerusakan | 50 = retak besar & dinding miring | 100 = bangunan hancur total"
    )
    lbl_k = "Ringan" if ker_v <= 35 else ("Sedang" if ker_v <= 65 else "Berat")
    cls_k = "ringan" if ker_v <= 35 else ("sedang" if ker_v <= 65 else "berat")
    st.markdown(f'Nilai: **{ker_v}** &nbsp;<span class="badge badge-{cls_k}">{lbl_k}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    gen_v = st.slider(
        "🌊 Tinggi Genangan Air (cm)",
        min_value=0, max_value=200, value=60, step=1,
        help="0 = tidak ada genangan | 50 cm = setinggi lutut | 120 cm = setinggi dada | 200 = rumah terendam total"
    )
    lbl_g = "Rendah" if gen_v <= 35 else ("Sedang" if gen_v <= 90 else "Tinggi")
    cls_g = "ringan" if gen_v <= 35 else ("sedang" if gen_v <= 90 else "berat")
    st.markdown(f'Nilai: **{gen_v} cm** &nbsp;<span class="badge badge-{cls_g}">{lbl_g}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    aks_v = st.slider(
        "🚑 Jarak ke Fasilitas Darurat (km)",
        min_value=0.0, max_value=15.0, value=3.0, step=0.1,
        help="0 km = posko/puskesmas di depan rumah | 5 km = ~10 menit berkendara | 15 km = sangat jauh dari bantuan"
    )
    lbl_a = "Dekat" if aks_v <= 3 else ("Sedang" if aks_v <= 8 else "Jauh")
    cls_a = "ringan" if aks_v <= 3 else ("sedang" if aks_v <= 8 else "berat")
    st.markdown(f'Nilai: **{aks_v:.1f} km** &nbsp;<span class="badge badge-{cls_a}">{lbl_a}</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        hitung = st.button("⚡ Hitung Kelayakan", use_container_width=True)
    with col_btn2:
        reset_riwayat = st.button("🗑️ Reset Riwayat", use_container_width=True)

    if reset_riwayat:
        st.session_state.riwayat = []
        st.rerun()

# VALIDASI
warnings_list = validate_input(ker_v, gen_v, aks_v)
for warn in warnings_list:
    st.markdown(f'<div class="warning-box">{warn}</div>', unsafe_allow_html=True)

# COMPUTE
if hitung:
    with st.spinner("📊 Sedang menganalisis kondisi rumah..."):
        import time
        time.sleep(0.5)
        score, active_rules, agg_data, memberships = infer(ker_v, gen_v, aks_v)
        km, gm, am = memberships

        st.session_state.hasil_dihitung = True
        st.session_state.score = score
        st.session_state.active_rules = active_rules
        st.session_state.agg_data = agg_data
        st.session_state.memberships = memberships
        st.session_state.km = km
        st.session_state.gm = gm
        st.session_state.am = am

        if score >= 65:
            status_riwayat = "Layak"
            icon_riwayat = "🟢"
        elif score >= 35:
            status_riwayat = "Layak Bersyarat"
            icon_riwayat = "🟡"
        else:
            status_riwayat = "Tidak Layak"
            icon_riwayat = "🔴"

        st.session_state.riwayat.insert(0, {
            'waktu': datetime.now().strftime("%d/%m %H:%M:%S"),
            'kerusakan': ker_v,
            'genangan': gen_v,
            'akses': aks_v,
            'score': score,
            'status': status_riwayat,
            'icon': icon_riwayat
        })

        if len(st.session_state.riwayat) > 10:
            st.session_state.riwayat = st.session_state.riwayat[:10]

# TAMPILKAN HASIL
if st.session_state.hasil_dihitung:
    score = st.session_state.score
    active_rules = st.session_state.active_rules
    agg_data = st.session_state.agg_data
    km = st.session_state.km
    gm = st.session_state.gm
    am = st.session_state.am

    if score >= 65:
        st_class, st_icon, st_text = "result-layak", "🟢", "LAYAK HUNI"
        rekomen_singkat = "✅ Rumah aman untuk ditempati."
    elif score >= 35:
        st_class, st_icon, st_text = "result-bersyarat", "🟡", "LAYAK BERSYARAT"
        rekomen_singkat = "⚠️ Dapat ditempati dengan syarat."
    else:
        st_class, st_icon, st_text = "result-tidak", "🔴", "TIDAK LAYAK HUNI"
        rekomen_singkat = "🚨 Segera evakuasi!"

    with col_r:
        st.markdown(f"""
        <div class="{st_class}">
            <div style="font-size:2rem">{st_icon}</div>
            <div class="result-score">{score:.1f}<span style="font-size:1rem;font-weight:400"> / 100</span></div>
            <div class="result-label">{st_text}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(rekomen_singkat)
        st.markdown(get_detailed_rekomendasi(score, ker_v, gen_v, aks_v))

        hasil_text = f"""
HASIL PENILAIAN KELAYAKAN RUMAH PASCA BENCANA
================================================
Waktu: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Parameter Input:
- Kerusakan Struktur: {ker_v} ({lbl_k})
- Tinggi Genangan Air: {gen_v} cm ({lbl_g})
- Jarak ke Fasilitas Darurat: {aks_v:.1f} km ({lbl_a})

Hasil Penilaian:
- Skor Kelayakan: {score:.1f} / 100
- Status: {st_text}

Rekomendasi:
{get_detailed_rekomendasi(score, ker_v, gen_v, aks_v)}

================================================
Sistem Pakar Fuzzy Mamdani - Centroid Defuzzification
"""

        st.download_button(
            label="📥 Download Hasil Penilaian (TXT)",
            data=hasil_text,
            file_name=f"hasil_kelayakan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown('<div class="card"><div class="card-title">📈 Derajat Keanggotaan Input</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**🏗️ Kerusakan Struktur**")
            for lbl, val, col in [("Ringan", km['ringan'], '#28a745'), ("Sedang", km['sedang'], '#856404'), ("Berat", km['berat'], '#dc3545')]:
                st.markdown(f'<div class="mem-box"><div class="mem-val" style="color:{col}">{val:.3f}</div><div class="mem-lbl">{lbl}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown("**🌊 Genangan Air**")
            for lbl, val, col in [("Rendah", gm['rendah'], '#28a745'), ("Sedang", gm['sedang'], '#856404'), ("Tinggi", gm['tinggi'], '#dc3545')]:
                st.markdown(f'<div class="mem-box"><div class="mem-val" style="color:{col}">{val:.3f}</div><div class="mem-lbl">{lbl}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown("**🚑 Akses Fasilitas**")
            for lbl, val, col in [("Dekat", am['dekat'], '#28a745'), ("Sedang", am['sedang'], '#856404'), ("Jauh", am['jauh'], '#dc3545')]:
                st.markdown(f'<div class="mem-box"><div class="mem-val" style="color:{col}">{val:.3f}</div><div class="mem-lbl">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        confidence = score
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🎯 Tingkat Keyakinan Sistem</div>
            <div style="font-size:1.2rem; font-weight:700; margin-bottom:0.3rem">{confidence:.1f}%</div>
            <div class="confidence-bar">
                <div style="background: linear-gradient(90deg, #28a745, #ffc107, #dc3545); width:{confidence}%; height:6px; border-radius:10px;"></div>
            </div>
            <p style="font-size:0.75rem; color:#6c757d; margin-top:0.3rem">
                {'Sangat Yakin' if confidence >= 70 else 'Cukup Yakin' if confidence >= 40 else 'Kurang Yakin'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ATURAN AKTIF
    st.markdown("---")
    st.markdown("### 📋 Aturan Fuzzy yang Aktif")

    if active_rules:
        cols = st.columns(2)
        for i, (rno, label, strength, out) in enumerate(active_rules):
            css = 'rule-tidak' if out == 'tidak' else ('rule-bersyarat' if out == 'bersyarat' else 'rule-layak')
            with cols[i % 2]:
                st.markdown(
                    f'<div class="rule-item {css}"><b>{label}</b>'
                    f'<span style="float:right;font-weight:700">{strength:.3f}</span></div>',
                    unsafe_allow_html=True
                )
    else:
        st.warning("Tidak ada aturan yang teraktivasi.")

    st.caption(f"✅ {len(active_rules)} aturan aktif  |  ⬜ {len(RULES)-len(active_rules)} tidak aktif  |  Total: 16 aturan")

    # GRAFIK
    with st.expander("📈 Lihat Grafik Membership Function & Defuzzifikasi", expanded=False):
        st.markdown("""
        <div class="info-box">
        💡 <b>Cara membaca grafik:</b> Garis putus-putus biru adalah nilai input Anda.
        Setiap kurva berwarna mewakili satu kategori —
        <span style="color:#28a745;font-weight:600">hijau = baik/ringan/dekat</span>,
        <span style="color:#856404;font-weight:600">kuning = sedang</span>,
        <span style="color:#dc3545;font-weight:600">merah = buruk/berat/jauh</span>.
        Semakin tinggi kurva di titik input Anda, semakin kuat sistem mengenali input tersebut sebagai kategori itu.
        </div>
        """, unsafe_allow_html=True)

        fig_mf = plot_mf(ker_v, gen_v, aks_v)
        st.pyplot(fig_mf, use_container_width=True)
        plt.close(fig_mf)

        st.markdown("---")
        st.markdown("**Proses Defuzzifikasi** — area arsiran adalah hasil agregasi semua aturan aktif. Garis biru putus-putus adalah nilai akhir skor (centroid/COG).")
        fig_def = plot_defuzz(score, agg_data)
        st.pyplot(fig_def, use_container_width=True)
        plt.close(fig_def)

else:
    with col_r:
        st.markdown("""
        <div class="card" style="text-align:center; min-height:400px; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:3.5rem; margin-bottom:1rem">👈</div>
            <h3>Silakan Masukkan Data</h3>
            <p style="color:#6c757d">Gunakan slider di sebelah kiri untuk memasukkan data kondisi rumah, lalu klik tombol <b>Hitung Kelayakan</b>.</p>
        </div>
        """, unsafe_allow_html=True)

# RIWAYAT (History)
if st.session_state.riwayat:
    st.markdown("---")
    st.markdown("### 🕒 Riwayat Penilaian")
    
    # Display as a horizontal scrolling area or simple list
    cols_hist = st.columns(len(st.session_state.riwayat) if len(st.session_state.riwayat) < 4 else 4)
    for i, item in enumerate(st.session_state.riwayat[:8]): # Show last 8
        with cols_hist[i % 4]:
            css_h = 'history-tidak' if item['status'] == 'Tidak Layak' else ('history-bersyarat' if item['status'] == 'Layak Bersyarat' else 'history-layak')
            st.markdown(f"""
            <div class="history-item {css_h}">
                <div style="font-weight:700; font-size:0.9rem">{item['icon']} {item['status']}</div>
                <div style="color:#6c757d; font-size:0.7rem; margin-bottom:5px">{item['waktu']}</div>
                <div style="font-size:0.75rem">
                    🏗️ {item['kerusakan']} | 🌊 {item['genangan']}cm | 🚑 {item['akses']}km
                </div>
                <div style="font-weight:700; margin-top:3px; color:#0f3460">Skor: {item['score']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
    Sistem Penilaian Kelayakan Hunian Pasca Bencana &copy; 2024 · Dibuat dengan Streamlit & Scikit-Fuzzy
</div>
""", unsafe_allow_html=True)
