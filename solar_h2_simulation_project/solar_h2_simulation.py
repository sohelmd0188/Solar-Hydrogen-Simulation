
"""
Solar–Hydrogen Techno-Economic Simulation Dashboard
Generated for user's research. Streamlit app.
Run with: streamlit run solar_h2_simulation.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# ----------------------------
# Constants (paper-derived defaults)
# ----------------------------
LHV_H2_kwh_per_kg = 33.33         # kWh/kg
DEFAULT_ETA_EL = 0.80
DEFAULT_ETA_FC = 0.50
DEFAULT_PV_MWp = 6.25
DEFAULT_PEL_KW = 5000
DEFAULT_STORAGE_KG = 700
DEFAULT_FC_KW = 2000
DEFAULT_PSH = 5.0
DEFAULT_DERATE = 0.80
DEFAULT_MH2_MONTHLY = 17400
DEFAULT_MH2_ANNUAL = DEFAULT_MH2_MONTHLY * 12

# ----------------------------
# Core Equations
# ----------------------------
def E_el_annual_kwh(mH2_annual, LHV=LHV_H2_kwh_per_kg, eta=DEFAULT_ETA_EL):
    return mH2_annual * LHV / eta

def Pel_rated_kw(Eel_annual_kwh, PSH_daily=DEFAULT_PSH):
    return Eel_annual_kwh / (PSH_daily * 365)

def PV_capacity_MWp_from_Eel(Eel_annual_kwh, PSH_daily=DEFAULT_PSH, derate=DEFAULT_DERATE):
    Eel_daily = Eel_annual_kwh / 365
    return Eel_daily / (PSH_daily * 1000 * derate)

def hydrogen_production_rate_kg_per_h(Pel_rated_kW, eta=DEFAULT_ETA_EL, LHV=LHV_H2_kwh_per_kg):
    return (Pel_rated_kW * eta) / LHV

def storage_volume_m3(storage_kg, density_kg_per_m3=23.1):
    return storage_kg / density_kg_per_m3

def capital_recovery_factor(r, n):
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def LCOH_simple(CAPEX, OPEX, revenue, mH2_annual, r=0.08, n=20):
    crf = capital_recovery_factor(r, n)
    return ((CAPEX * crf) + OPEX - revenue) / max(1.0, mH2_annual)

# ----------------------------
# Streamlit UI Layout
# ----------------------------
st.set_page_config(page_title="Solar–Hydrogen Simulation", layout="wide", initial_sidebar_state="expanded")
st.title("Solar–Hydrogen Techno-Economic Simulation Dashboard")
st.markdown("Interactive simulation based on the user's research paper. Use the left panel to change inputs. Outputs, plots and tables appear on the right.")

# Sidebar Inputs
st.sidebar.header("Input Controls")

season = st.sidebar.selectbox("Season Type", ["Annual Average", "Hot Season", "Cold Season"])
mH2_annual = st.sidebar.number_input("Annual H₂ Production (kg/yr)", min_value=10000, max_value=1_000_000, value=int(DEFAULT_MH2_ANNUAL), step=1000)
eta_el = st.sidebar.number_input("Electrolyzer Efficiency η_el", min_value=0.50, max_value=0.95, value=DEFAULT_ETA_EL, step=0.01, format="%.2f")
eta_fc = st.sidebar.number_input("Fuel Cell Efficiency η_fc", min_value=0.30, max_value=0.90, value=DEFAULT_ETA_FC, step=0.01, format="%.2f")
PSH_daily = st.sidebar.number_input("Peak Sun Hours (h/day)", min_value=1.0, max_value=10.0, value=DEFAULT_PSH, step=0.1, format="%.1f")
derate = st.sidebar.number_input("PV Derating Factor", min_value=0.5, max_value=1.0, value=DEFAULT_DERATE, step=0.01, format="%.2f")
storage_kg = st.sidebar.number_input("H₂ Storage Capacity (kg)", min_value=0, max_value=10_000, value=DEFAULT_STORAGE_KG, step=10)

# Economic inputs
st.sidebar.markdown("### Economic Parameters")
elec_capex = st.sidebar.number_input("Electrolyzer CAPEX (USD/kW)", min_value=100, max_value=5000, value=1050, step=10)
pv_capex = st.sidebar.number_input("PV CAPEX (USD/kW)", min_value=100, max_value=5000, value=700, step=10)
storage_capex = st.sidebar.number_input("Storage CAPEX (USD/kg)", min_value=10, max_value=2000, value=650, step=10)
OPEX = st.sidebar.number_input("Annual OPEX (USD/yr)", min_value=0, max_value=1_000_000, value=60000, step=1000)
revenue = st.sidebar.number_input("Byproduct Revenue (USD/yr)", min_value=0, max_value=1_000_000, value=160000, step=1000)
r = st.sidebar.number_input("Discount Rate r", min_value=0.01, max_value=0.25, value=0.08, step=0.01, format="%.2f")
n = st.sidebar.number_input("Project Lifetime (years)", min_value=1, max_value=50, value=20, step=1)

# Optional Monte-Carlo
st.sidebar.markdown("### Analysis Options")
run_monte = st.sidebar.checkbox("Run Monte-Carlo LCOH (quick, 2000 iters)", value=False)

# ----------------------------
# Derived calculations
# ----------------------------
Eel_annual = E_el_annual_kwh(mH2_annual, LHV_H2_kwh_per_kg, eta_el)
Pel_rated = Pel_rated_kw(Eel_annual, PSH_daily)
PV_capacity = PV_capacity_MWp_from_Eel(Eel_annual, PSH_daily, derate)
H2_rate = hydrogen_production_rate_kg_per_h(Pel_rated, eta_el)
storage_vol = storage_volume_m3(storage_kg)
CAPEX_total = elec_capex * Pel_rated + pv_capex * PV_capacity * 1000 + storage_capex * storage_kg
LCOH = LCOH_simple(CAPEX_total, OPEX, revenue, mH2_annual, r, int(n))

# ----------------------------
# Layout: Left constants/info + Right outputs
# ----------------------------
col1, col2 = st.columns([1.0, 2.0])

with col1:
    st.subheader("Key Constants & Presets")
    const_table = pd.DataFrame({
        "Parameter": ["LHV of H₂ (kWh/kg)", "Default Electrolyzer η", "Default Fuel Cell η", "Default PSH (h/day)", "Default PV derate"],
        "Value": [f"{LHV_H2_kwh_per_kg}", f"{DEFAULT_ETA_EL*100:.1f}%", f"{DEFAULT_ETA_FC*100:.1f}%", f"{DEFAULT_PSH}", f"{DEFAULT_DERATE*100:.0f}%"]
    })
    st.table(const_table)

    st.subheader("Economic Summary")
    eco_table = pd.DataFrame({
        "Item": ["Estimated Total CAPEX (USD)", "Annual OPEX (USD/yr)", "Byproduct Revenue (USD/yr)", "Estimated LCOH (USD/kg)"],
        "Value": [f"{CAPEX_total:,.0f}", f"{OPEX:,.0f}", f"{revenue:,.0f}", f"{LCOH:.2f}"]
    })
    st.table(eco_table)

    if st.button("Export current results to CSV"):
        out_df = pd.DataFrame({
            "metric": ["mH2_annual_kg", "E_el_annual_kWh", "Pel_rated_kW", "PV_capacity_MWp", "H2_rate_kgph", "storage_kg", "storage_m3", "CAPEX_USD", "LCOH_USD_per_kg"],
            "value": [mH2_annual, Eel_annual, Pel_rated, PV_capacity, H2_rate, storage_kg, storage_vol, CAPEX_total, LCOH]
        })
        csv_bytes = out_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download results CSV", data=csv_bytes, file_name="solar_h2_results.csv", mime="text/csv")

with col2:
    st.subheader(f"Simulation Results — {season}")
    results = {
        "Annual H₂ Production (kg)": mH2_annual,
        "E_el,annual (kWh)": Eel_annual,
        "Electrolyzer Rated Power (MW)": Pel_rated / 1000.0,
        "PV Capacity (MWp)": PV_capacity,
        "H₂ Rate (kg/h)": H2_rate,
        "Storage Volume (m³)": storage_vol,
        "Estimated LCOH (USD/kg)": LCOH
    }
    st.dataframe(pd.DataFrame(results, index=["Value"]).T)

    # Plots
    st.markdown("### Dynamic Relationships")
    mvals = np.linspace(mH2_annual * 0.5, mH2_annual * 1.5, 100)
    Evals = [E_el_annual_kwh(m, LHV_H2_kwh_per_kg, eta_el) for m in mvals]
    Pel_vals = [Pel_rated_kw(E_el_annual_kwh(m, LHV_H2_kwh_per_kg, eta_el), PSH_daily) for m in mvals]
    PVvals = [PV_capacity_MWp_from_Eel(E_el_annual_kwh(m, LHV_H2_kwh_per_kg, eta_el), PSH_daily, derate) for m in mvals]
    H2rate_vals = [hydrogen_production_rate_kg_per_h(p, eta_el) for p in Pel_vals]

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(mvals / 1000.0, np.array(Evals) / 1e6)
    axs[0].scatter(mH2_annual / 1000.0, Eel_annual / 1e6, color='red')
    axs[0].set_xlabel('H₂ Annual (×10³ kg)')
    axs[0].set_ylabel('E_el,annual (GWh)')
    axs[0].set_title('E_el,annual vs H₂ Production')

    axs[1].plot(np.array(Pel_vals) / 1000.0, PVvals)
    axs[1].scatter(Pel_rated / 1000.0, PV_capacity, color='red')
    axs[1].set_xlabel('Electrolyzer Power (MW)')
    axs[1].set_ylabel('PV Capacity (MWp)')
    axs[1].set_title('PV vs Electrolyzer')

    axs[2].plot(np.array(Pel_vals) / 1000.0, H2rate_vals)
    axs[2].scatter(Pel_rated / 1000.0, H2_rate, color='red')
    axs[2].set_xlabel('Electrolyzer Power (MW)')
    axs[2].set_ylabel('H₂ Rate (kg/h)')
    axs[2].set_title('H₂ Rate vs Electrolyzer Power')

    plt.tight_layout()
    st.pyplot(fig)

    # =========================================================
# 🌍 Monthly CO₂ Mitigation & Cost-Profit Analysis Section
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

st.markdown("## 🌿 Monthly CO₂ Mitigation & Cost–Profit Analysis")

months = [
    "Apr'24", "May'24", "Jun'24", "Jul'24", "Aug'24",
    "Sep'24", "Oct'24", "Nov'24", "Dec'24",
    "Jan'25", "Feb'25", "Mar'25", "Apr'25"
]

# Input block
selected_month = st.selectbox("Select Month", months)

# Dummy data
grid_import_mwh = [0, 37.0, 0, 14.5, 0, 34.0, 22.0, 7.0, 0, 0, 0, 0, 29.5]
grid_export_mwh = [8.0, 0, 2.0, 0, 33.5, 0, 0, 0, 33.5, 48.5, 41.0, 15.5, 0]
diesel_reduction = [112560] * 13
factor = 710  # kg CO₂/MWh

grid_export_reduction = [val * factor for val in grid_export_mwh]
grid_import_emissions = [val * factor for val in grid_import_mwh]
total_mitigation = (
    np.array(diesel_reduction) + np.array(grid_export_reduction) - np.array(grid_import_emissions)
)

i = months.index(selected_month)
diesel = diesel_reduction[i]
export_red = grid_export_reduction[i]
import_em = grid_import_emissions[i]
total = total_mitigation[i]

fig1, ax1 = plt.subplots(figsize=(9, 6))
categories = ["Diesel Reduction", "Grid Export Reduction", "Grid Import Emission"]
values = [diesel, export_red, -import_em]
colors = ["#3b82f6", "#22c55e", "#ef4444"]

# Bars
bars = ax1.bar(categories, values, color=colors, edgecolor='black', alpha=0.85)

# Add value labels on each bar
for bar, val in zip(bars, values):
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        val + (0.02 * max(values + [total])),  # slight offset
        f"{val/1000:.1f}K",
        ha='center', va='bottom',
        fontsize=12, fontweight='bold',
        color='black'
    )

fig1, ax1 = plt.subplots(figsize=(9, 6))

# Categories & Data
categories = ["Diesel Reduction", "Grid Export Reduction", "Grid Import Emission"]
values = [diesel, export_red, -import_em]
colors = ["#3b82f6", "#22c55e", "#ef4444"]

# Bars
bars = ax1.bar(categories, values, color=colors, edgecolor='black', alpha=0.85)

# === Add Value Labels Clearly ===
for bar, val in zip(bars, values):
    # Position text above positive bars and below negative bars
    y_pos = val + (0.05 * max(values)) if val >= 0 else val - (0.08 * abs(min(values)))
    va = 'bottom' if val >= 0 else 'top'

    ax1.text(
        bar.get_x() + bar.get_width()/2,
        y_pos,
        f"{val/1000:.1f}K",
        ha='center', va=va,
        fontsize=13, fontweight='bold',
        color='black'
    )

# === Total Mitigation Line & Label ===
ax1.axhline(total, color='black', linestyle='--', linewidth=3, label="Total Mitigation")

# Keep label always visible slightly above line
y_label = total + (0.06 * max(values))
ax1.text(
    1.5, y_label,
    f"Total Mitigation: {total/1000:.1f}K kg CO₂",
    ha='center', va='bottom',
    fontsize=11, fontweight='bold', color='black'
)

# === Titles & Styling ===
ax1.set_title(f"CO₂ Mitigation – {selected_month}", fontsize=18, fontweight='bold')
ax1.set_ylabel("CO₂ (kg)", fontsize=14)
ax1.grid(alpha=0.3)
ax1.legend()

# Add a small padding for better label visibility
ax1.set_ylim(min(values) * 1.3, max(values + [total]) * 1.3)

st.pyplot(fig1)


# Cost-Profit Table (Dummy values)
cost_profit_data = {
    "Month": months,
    "Hydrogen Produced (kg)": [17000, 16800, 17500, 17200, 17800, 17600, 16900, 17300, 17100, 16500, 16400, 16800, 17400],
    "Grid Import (MWh)": grid_import_mwh,
    "Grid Export (MWh)": grid_export_mwh,
    "Cost (USD)": [1200, 1350, 1180, 1220, 1250, 1400, 1300, 1190, 1210, 1150, 1120, 1180, 1270],
    "Profit (USD)": [1800, 1500, 1900, 1700, 2000, 1600, 1750, 1800, 2100, 1950, 1850, 2000, 2050]
}
df_cost = pd.DataFrame(cost_profit_data)
st.markdown("### 💰 Monthly Cost–Profit Summary")
st.dataframe(df_cost)


# Monte-Carlo quick
if run_monte:
    st.markdown("### Monte-Carlo LCOH (quick)")
    import numpy as _np
    iters = 2000
    _np.random.seed(0)
    elec_capex_samp = _np.random.normal(elec_capex, elec_capex*0.15, iters)
    pv_capex_samp = _np.random.normal(pv_capex, pv_capex*0.15, iters)
    storage_capex_samp = _np.random.normal(storage_capex, storage_capex*0.15, iters)
    OPEX_samp = _np.random.normal(OPEX, OPEX*0.15, iters)
    revenue_samp = _np.random.normal(revenue, max(1,revenue*0.2), iters)
    capex_total_samp = elec_capex_samp * Pel_rated + pv_capex_samp * PV_capacity * 1000 + storage_capex_samp * storage_kg
    lcoh_samp = [LCOH_simple(capex_total_samp[i], OPEX_samp[i], revenue_samp[i], mH2_annual, r, int(n)) for i in range(iters)]
    lcoh_arr = _np.array(lcoh_samp)
    st.write("Monte-Carlo results (mean ± std):", float(lcoh_arr.mean()), "±", float(lcoh_arr.std()))
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.hist(lcoh_arr, bins=60)
    ax2.set_xlabel("LCOH (USD/kg)")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)
    # Download Monte-Carlo CSV
    mc_df = pd.DataFrame({"LCOH_USD_per_kg": lcoh_arr})
    csv_bytes = mc_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Monte-Carlo LCOH CSV", data=csv_bytes, file_name="montecarlo_lcoh.csv", mime="text/csv")


    # =========================================================
# ⚡ Energy Management Simulation (EMS)
# =========================================================
st.markdown("## ⚙️ Energy Management Simulation (EMS)")

if st.checkbox("Run EMS Simulation"):
    months_ems = ["Apr-24","May-24","Jun-24","Jul-24","Aug-24","Sep-24",
                  "Oct-24","Nov-24","Dec-24","Jan-25","Feb-25","Mar-25","Apr-25"]
    selected_ems = st.selectbox("Select Month for EMS", months_ems)

    # 🔹 EMS core calculation (simplified from your provided code)
    import numpy as np
    PV_kW = 6250; PSH = 5; DERATING = 0.8
    ELECTROLYZER_RATED = 5000; ELECTROLYZER_EFF = 0.80
    FC_CAPACITY = 2000; FC_EFF = 0.50; LHV_H2 = 33.33
    MAX_STORAGE_KG = 700; RESERVE_KG = 200; CRITICAL_KG = 100
    days = 30
    hours = days * 24

    SOLAR_PATTERN = np.array([0,0,0,0,0,500,2500,4500,6000,6250,6000,4500,3000,2000,1000,300,0,0,0,0,0,0,0,0])
    LOAD_PATTERN  = np.array([1800,1600,1400,1300,1500,2000,2800,3200,3500,3300,3100,2900,2700,2500,2400,2300,2200,2100,2000,1900,1800,1800,1800,1800])
    solar = np.tile(SOLAR_PATTERN, days)
    load = np.tile(LOAD_PATTERN, days)
    solar = solar.astype(float)
    solar *= (PV_kW * PSH * DERATING * days) / np.sum(solar)


    H2 = 540 * LHV_H2
    H2_series, ELZ_series, FC_series, Gimp_series, Gexp_series = [], [], [], [], []

    for t in range(hours):
        pv = solar[t]; ld = load[t]
        elz = fc = gimp = gexp = 0
        if pv >= ld:
            surplus = pv - ld
            elz = min(surplus, ELECTROLYZER_RATED)
            H2 += elz * ELECTROLYZER_EFF
            if H2 > MAX_STORAGE_KG * LHV_H2:
                H2 = MAX_STORAGE_KG * LHV_H2
            if surplus - elz > 100:
                gexp = surplus - elz
        else:
            deficit = ld - pv
            if H2 > RESERVE_KG * LHV_H2:
                fc = min(deficit, FC_CAPACITY)
                H2 -= fc / FC_EFF
                deficit -= fc
            if deficit > 0:
                gimp = deficit

        ELZ_series.append(elz); FC_series.append(fc)
        Gimp_series.append(gimp); Gexp_series.append(gexp)
        H2_series.append(H2 / LHV_H2)

    time = np.arange(hours)

    # Plot 1: PV vs Load
    fig_pv, ax_pv = plt.subplots(figsize=(10, 4))
    ax_pv.plot(time, solar, color="orange", label="PV Generation (kW)")
    ax_pv.plot(time, load, color="blue", label="Load (kW)")
    ax_pv.set_title(f"{selected_ems}: PV vs Load", fontweight='bold')
    ax_pv.legend(); ax_pv.grid(alpha=0.3)
    st.pyplot(fig_pv)

    # Plot 2: EMS Dispatch
    fig_ems, ax_ems = plt.subplots(figsize=(10, 4))
    ax_ems.plot(time, ELZ_series, label="Electrolyzer (kW)", color="green")
    ax_ems.plot(time, FC_series, label="Fuel Cell (kW)", color="red")
    ax_ems.plot(time, Gimp_series, label="Grid Import (kW)", color="purple")
    ax_ems.plot(time, Gexp_series, label="Grid Export (kW)", color="cyan")
    ax_ems.set_title(f"{selected_ems}: EMS Dispatch", fontweight='bold')
    ax_ems.legend(); ax_ems.grid(alpha=0.3)
    st.pyplot(fig_ems)

    # Plot 3: H₂ Storage Dynamics
    fig_h2, ax_h2 = plt.subplots(figsize=(10, 4))
    ax_h2.plot(time, H2_series, color="darkgreen", linewidth=2, label="H₂ Storage (kg)")
    ax_h2.axhline(MAX_STORAGE_KG, color="red", linestyle="--", label="Max 700 kg")
    ax_h2.axhline(RESERVE_KG, color="orange", linestyle="--", label="Reserve 200 kg")
    ax_h2.axhline(CRITICAL_KG, color="brown", linestyle="--", label="Critical 100 kg")
    ax_h2.set_title(f"{selected_ems}: Hydrogen Storage Dynamics", fontweight='bold')
    ax_h2.legend(); ax_h2.grid(alpha=0.3)
    st.pyplot(fig_h2)


st.markdown("---")
st.markdown("Generated project: Solar–Hydrogen Simulation. Ensure required packages installed: streamlit, numpy, pandas, matplotlib.")
