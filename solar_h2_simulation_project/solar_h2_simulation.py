
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

def NPV_simple(annual_cashflow, CAPEX, r, n):
    return sum(annual_cashflow / ((1 + r) ** t) for t in range(1, n + 1)) - CAPEX


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
elec_capex = st.sidebar.number_input("Electrolyzer CAPEX (USD/kW)", min_value=100, max_value=5000, value=1400, step=10)
pv_capex = st.sidebar.number_input("PV CAPEX (USD/kW)", min_value=100, max_value=5000, value=700, step=10)
fc_capex = st.sidebar.number_input( "Fuel Cell CAPEX (USD/kW)", min_value=100,max_value=5000,value=1200,step=50)
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
CAPEX_total = elec_capex * Pel_rated + pv_capex * PV_capacity * 1000 + storage_capex * storage_kg + fc_capex * (DEFAULT_FC_KW)
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

    #Sensitivity Analysis (LCOH & NPV)
    st.markdown("### Sensitivity Analysis (LCOH & NPV)")

    capex_factors = np.linspace(0.7, 1.3, 20)
    lcoh_sens = [
    LCOH_simple(CAPEX_total * f, OPEX, revenue, mH2_annual, r, int(n))
    for f in capex_factors
    ]

    fig, ax = plt.subplots()
    ax.plot(capex_factors * 100, lcoh_sens)
    ax.set_xlabel("CAPEX Variation (%)")
    ax.set_ylabel("LCOH (USD/kg)")
    ax.set_title("Sensitivity of LCOH to CAPEX")
    ax.grid(True)
    st.pyplot(fig)


=======
    annual_cashflow = revenue - OPEX
    r_vals = np.linspace(0.03, 0.15, 20)
    npv_vals = [NPV_simple(annual_cashflow, CAPEX_total, rv, int(n)) for rv in r_vals]

    fig2, ax2 = plt.subplots()
    ax2.plot(r_vals * 100, npv_vals)
    ax2.set_xlabel("Discount Rate (%)")
    ax2.set_ylabel("NPV (USD)")
    ax2.set_title("Sensitivity of NPV to Discount Rate")
    ax2.grid(True)
    st.pyplot(fig2)


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

    #Energy Management Simulation  
    st.markdown("### Energy Management Simulation (Monthly Aggregated)")

    uploaded_file = st.file_uploader(
    "Upload Monthly Excel/CSV (Month, Electricity_Demand_kWh, Solar_Generation_kWh)",
    type=["csv", "xlsx"]
    )

    if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    df["Grid_Import_kWh"] = np.maximum(
        df["Electricity_Demand_kWh"] - df["Solar_Generation_kWh"], 0
    )
    df["Grid_Export_kWh"] = np.maximum(
        df["Solar_Generation_kWh"] - df["Electricity_Demand_kWh"], 0
    )

    st.dataframe(df)


    #Monthly CO₂ Mitigation + Cost–Profit Plot
    grid_emission_factor = 0.67  # kg CO2/kWh

    df["CO2_Emitted_kg"] = df["Grid_Import_kWh"] * grid_emission_factor
    df["CO2_Avoided_kg"] = df["Grid_Export_kWh"] * grid_emission_factor

    df["Import_Cost_USD"] = df["Grid_Import_kWh"] * 0.12
    df["Export_Revenue_USD"] = df["Grid_Export_kWh"] * 0.08
    df["Net_Profit_USD"] = df["Export_Revenue_USD"] - df["Import_Cost_USD"]

    fig, ax = plt.subplots()
    ax.bar(df["Month"], df["Net_Profit_USD"])
    ax.set_title("Monthly Net Cost–Profit")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot(df["Month"], df["CO2_Avoided_kg"], label="CO₂ Avoided")
    ax2.plot(df["Month"], df["CO2_Emitted_kg"], label="CO₂ Emitted")
    ax2.legend()
    st.pyplot(fig2)


    #Documentation 
    st.markdown("### Documentation / Important Documents")

    st.markdown("""
    #### Methodology
    (Add step-by-step methodology later)

    #### Core Equations
    - Electrolyzer energy balance  
    - PV sizing  
    - LCOH  
    - NPV  

    #### Assumptions   
    - Monthly aggregated data  
    - Fixed efficiencies  
    - Bangladesh grid emission factor  

    ## References
    (Add journal papers, standards, datasets)
    """)


st.markdown("---")
st.markdown("Generated project: Solar–Hydrogen Simulation. Ensure required packages installed: streamlit, numpy, pandas, matplotlib.This model is Developed by Mohammad Sohel")

