# app.py
from pathlib import Path
import json, sqlite3, time
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE = Path(__file__).parent
DATA_DIR = BASE / "app_data"
DATA_BUNDLE = DATA_DIR / "data_bundle.json"
H160_ITEMS = DATA_DIR / "h160_items.json"
DB_PATH = BASE / "inputs.db"

st.set_page_config(page_title="H160 Load & CG", layout="centered", initial_sidebar_state="expanded")

# --- helpers ---
def load_data():
    with open(DATA_BUNDLE) as f:
        data = json.load(f)
    with open(H160_ITEMS) as f:
        items = json.load(f)
    fuel_df = pd.DataFrame(data["fuel_table"])
    fuel_df = fuel_df.sort_values("Fuel_kg").reset_index(drop=True)
    return data, items, fuel_df

def interp_fuel(fuel_df, fuel_kg):
    # linear interpolation for arm and moment
    if fuel_kg <= float(fuel_df.iloc[0]["Fuel_kg"]):
        return float(fuel_df.iloc[0]["Arm_m"]), float(fuel_df.iloc[0]["Moment"])
    if fuel_kg >= float(fuel_df.iloc[-1]["Fuel_kg"]):
        return float(fuel_df.iloc[-1]["Arm_m"]), float(fuel_df.iloc[-1]["Moment"])
    lower = fuel_df[fuel_df["Fuel_kg"] <= fuel_kg].iloc[-1]
    upper = fuel_df[fuel_df["Fuel_kg"] >= fuel_kg].iloc[0]
    if lower["Fuel_kg"] == upper["Fuel_kg"]:
        return float(lower["Arm_m"]), float(lower["Moment"])
    frac = (fuel_kg - lower["Fuel_kg"]) / (upper["Fuel_kg"] - lower["Fuel_kg"])
    arm = float(lower["Arm_m"]) + frac * (float(upper["Arm_m"]) - float(lower["Arm_m"]))
    moment = float(lower["Moment"]) + frac * (float(upper["Moment"]) - float(lower["Moment"]))
    return arm, moment

def find_item(items, contains):
    for it in items:
        if isinstance(it["item"], str) and contains.lower() in it["item"].lower():
            return it
    return None

def compute_moments(inputs, items, fuel_arm, fuel_moment, fuel_kg):
    # Build table rows with arms and moments for long and lat
    rows = []
    total_long_moment = 0.0
    total_lat_moment = 0.0
    total_weight_payload = 0.0
    empty_weight = 0.0
    empty_long_moment = 0.0
    empty_lat_moment = 0.0

    # find empty weight item
    empty = find_item(items, "empty weight")
    if empty:
        empty_weight = float(empty.get("weight") or 0.0)
        empty_long_moment = float(empty.get("moment_long") or 0.0)
        empty_lat_moment = float(empty.get("moment_lat") or 0.0)

    # go through each input and compute moments using arms from items table
    for key, val in inputs.items():
        if key == "takeoff_fuel_kg": continue
        it = find_item(items, key)
        if it is None:
            # skip unknown names
            continue
        w = float(val or 0.0)
        arm_long = it.get("arm_long")
        arm_lat = it.get("arm_lat")
        arm_long = float(arm_long) if arm_long not in (None, "nan") else None
        arm_lat = float(arm_lat) if arm_lat not in (None, "nan") else 0.0
        moment_long = w * arm_long if arm_long is not None else None
        moment_lat = w * arm_lat if arm_lat is not None else None
        if moment_long is not None:
            total_long_moment += moment_long
        if moment_lat is not None:
            total_lat_moment += moment_lat
        total_weight_payload += w
        rows.append({
            "item": it["item"],
            "weight": w,
            "arm_long": arm_long,
            "moment_long": moment_long,
            "arm_lat": arm_lat,
            "moment_lat": moment_lat
        })

    # add empty weight to totals
    total_weight_zero_fuel = empty_weight + total_weight_payload
    total_long_moment_zero = (empty_long_moment or 0.0) + total_long_moment
    total_lat_moment_zero = (empty_lat_moment or 0.0) + total_lat_moment

    # fuel contribution
    fuel_kg = inputs.get("takeoff_fuel_kg", 0.0)
    fuel_arm_val, fuel_moment_val = fuel_arm, fuel_moment  # already computed externally

    return {
        "rows": rows,
        "empty_weight": empty_weight,
        "zero_weight": total_weight_zero_fuel,
        "zero_long_moment": total_long_moment_zero,
        "zero_lat_moment": total_lat_moment_zero,
        "payload_weight": total_weight_payload
    }

def cg_from(weight, moment):
    if weight == 0:
        return None
    return moment / weight

# --- persistence (simple sqlite) ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS inputs_history (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            pilot REAL,
            copilot REAL,
            pax1 REAL, pax2 REAL, pax3 REAL, pax4 REAL, pax5 REAL, pax6 REAL,
            cargo1 REAL, cargo2 REAL, cargo3 REAL,
            takeoff_fuel REAL,
            cg_long REAL, cg_lat REAL,
            total_weight REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_history(row):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO inputs_history (
            timestamp, pilot, copilot, pax1, pax2, pax3, pax4, pax5, pax6,
            cargo1, cargo2, cargo3, takeoff_fuel, cg_long, cg_lat, total_weight
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        row["timestamp"], row["pilot"], row["copilot"], row["pax1"], row["pax2"], row["pax3"], row["pax4"], row["pax5"], row["pax6"],
        row["cargo1"], row["cargo2"], row["cargo3"], row["takeoff_fuel"],
        row["cg_long"], row["cg_lat"], row["total_weight"]
    ))
    conn.commit()
    conn.close()

def load_history(limit=10):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM inputs_history ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

# --- app start ---
data, items, fuel_df = load_data()
init_db()

st.title("H160 Load Balance App")
st.markdown("Enter weights (kg) in the inputs. The app computes moments, CG (long & lat) and draws the LONG & LAT envelopes with the CG points (D=zero fuel, G=650 kg, J=takeoff fuel).")

# Build inputs
st.header("Mission Inputs")
col1, col2 = st.columns(2)
with col1:
    pilot = st.number_input("Pilot (kg)", value=float(find_item(items, "pilot")["weight"]), step=1.0)
    copilot = st.number_input("Co-Pilot (kg)", value=float(find_item(items, "co-pilot")["weight"]), step=1.0)
    pax1 = st.number_input("FWD Row Pax 1 (kg)", value=float(find_item(items, "FWD Row Pax 1")["weight"]), step=1.0)
    pax2 = st.number_input("FWD Row Pax 2 (kg)", value=float(find_item(items, "FWD Row Pax 2")["weight"]), step=1.0)
    pax3 = st.number_input("FWD Row Pax 3 (kg)", value=float(find_item(items, "FWD Row Pax 3")["weight"]), step=1.0)
with col2:
    pax4 = st.number_input("FWD Row Pax 4 (kg)", value=float(find_item(items, "FWD Row Pax 4")["weight"]), step=1.0)
    pax5 = st.number_input("AFT RH Pax 5 (kg)", value=float(find_item(items, "AFT RH Pax 5")["weight"]), step=1.0)
    pax6 = st.number_input("AFT LH Pax 6 (kg)", value=float(find_item(items, "AFT LH Pax 6")["weight"]), step=1.0)
    cargo1 = st.number_input("Cargo 1 (kg)", value=float(find_item(items, "Cargo 1")["weight"]), step=1.0)
    cargo2 = st.number_input("Cargo 2 (kg)", value=float(find_item(items, "Cargo 2")["weight"]), step=1.0)
    cargo3 = st.number_input("Cargo 3 (kg)", value=float(find_item(items, "Cargo 3")["weight"]), step=1.0)
st.write("---")
takeoff_fuel = st.number_input("Take-off Fuel (kg)", value=890.0, step=1.0)

# compute fuel arm/moment via interpolation
fuel_arm, fuel_moment = interp_fuel(fuel_df, takeoff_fuel)

# build inputs dict in the same keys the find_item helper expects
inputs = {
    "pilot": pilot,
    "co-pilot": copilot,
    "fwd row pax 1": pax1,
    "fwd row pax 2": pax2,
    "fwd row pax 3": pax3,
    "fwd row pax 4": pax4,
    "aft rh pax 5": pax5,
    "aft lh pax 6": pax6,
    "cargo 1": cargo1,
    "cargo 2": cargo2,
    "cargo 3": cargo3,
    "takeoff_fuel_kg": takeoff_fuel
}

# compute zero fuel totals (empty + payload)
cmp = compute_moments(inputs, items, fuel_arm, fuel_moment, takeoff_fuel)
empty_weight = cmp["empty_weight"]
zero_weight = cmp["zero_weight"]
zero_long_moment = cmp["zero_long_moment"]
zero_lat_moment = cmp["zero_lat_moment"]

# Points required:
# D (zero fuel) -> zero_weight, zero_long_moment, zero_lat_moment
# G (650 kg) -> zero_weight + 650, zero_long_moment + fuel_moment_650, ...
fuel_arm_650, fuel_moment_650 = interp_fuel(fuel_df, 650.0)
weight_D = zero_weight
moment_long_D = zero_long_moment
moment_lat_D = zero_lat_moment
cg_long_D = cg_from(weight_D, moment_long_D)
cg_lat_D = cg_from(weight_D, moment_lat_D)

weight_G = zero_weight + 650.0
moment_long_G = moment_long_D + fuel_moment_650
moment_lat_G = moment_lat_D + 0.0  # assume lateral fuel arm = 0
cg_long_G = cg_from(weight_G, moment_long_G)
cg_lat_G = cg_from(weight_G, moment_lat_G)

weight_J = zero_weight + takeoff_fuel
moment_long_J = moment_long_D + fuel_moment
moment_lat_J = moment_lat_D + 0.0
cg_long_J = cg_from(weight_J, moment_long_J)
cg_lat_J = cg_from(weight_J, moment_lat_J)

# display results table (compact)
st.header("Computed Results")
res_df = pd.DataFrame([
    {"case": "D (Zero Fuel)", "weight_kg": round(weight_D,2), "cg_long_m": round(cg_long_D,4) if cg_long_D else None, "cg_lat_m": round(cg_lat_D,4) if cg_lat_D else None},
    {"case": "G (650 kg Fuel)", "weight_kg": round(weight_G,2), "cg_long_m": round(cg_long_G,4) if cg_long_G else None, "cg_lat_m": round(cg_lat_G,4) if cg_lat_G else None},
    {"case": "J (Take-off Fuel)", "weight_kg": round(weight_J,2), "cg_long_m": round(cg_long_J,4) if cg_long_J else None, "cg_lat_m": round(cg_lat_J,4) if cg_lat_J else None},
])
st.table(res_df)

# store entry in DB
save_row = {
    "timestamp": datetime.utcnow().isoformat(),
    "pilot": pilot, "copilot": copilot, "pax1": pax1, "pax2": pax2, "pax3": pax3, "pax4": pax4, "pax5": pax5, "pax6": pax6,
    "cargo1": cargo1, "cargo2": cargo2, "cargo3": cargo3,
    "takeoff_fuel": takeoff_fuel,
    "cg_long": cg_long_J if cg_long_J else None,
    "cg_lat": cg_lat_J if cg_lat_J else None,
    "total_weight": weight_J
}
save_history(save_row)

# Plot LONG chart
st.header("Longitudinal CG Envelope")
long_env_x = data["long_envelope"]["x"]
long_env_y = data["long_envelope"]["y"]

fig_long = go.Figure()
fig_long.add_trace(go.Scatter(
    x=long_env_x, y=long_env_y, mode='lines', fill='toself', name='LONG envelope', hoverinfo='skip'
))
# limit line
lim_x = data["limit_line"]["x"]
lim_y = data["limit_line"]["y"]
# compute a horizontal limit line that spans the LONG envelope's X-range
limit_value = data["limit_line"]["y"][0]   # e.g. 6100
lim_x_long = [min(long_env_x), max(long_env_x)]
fig_long.add_trace(go.Scatter(
    x=lim_x_long,
    y=[limit_value, limit_value],
    mode='lines',
    name='Weight limit',
    line=dict(color='red', dash='dash')   # optional styling
))

# CG points (D,G,J)
fig_long.add_trace(go.Scatter(
    x=[cg_long_D, cg_long_G, cg_long_J],
    y=[weight_D, weight_G, weight_J],
    mode='lines+markers+text',
    line=dict(shape='linear'),            # optional but explicit
    marker=dict(size=8),
    text=["D","G","J"],
    textposition="top center",
    name="CG Points"
))

fig_long.update_layout(xaxis_title="Longitudinal Arm (m)", yaxis_title="Weight (kg)", height=450)
st.plotly_chart(fig_long, use_container_width=True)

# Plot LAT chart
st.header("Lateral CG Envelope")
lat_env_x = data["lat_envelope"]["x"]
lat_env_y = data["lat_envelope"]["y"]
fig_lat = go.Figure()
fig_lat.add_trace(go.Scatter(x=lat_env_x, y=lat_env_y, mode='lines', fill='toself', name='LAT envelope', hoverinfo='skip'))
fig_lat.add_trace(go.Scatter(x=lim_x, y=lim_y, mode='lines', name='Weight limit'))
fig_lat.add_trace(go.Scatter(x=[cg_lat_D, cg_lat_G, cg_lat_J],
                             y=[weight_D, weight_G, weight_J],
                             mode='lines+markers+text', marker=dict(size=8), line=dict(shape='linear'),
                             text=["D","G","J"], textposition="top center", name="CG Points"))
fig_lat.update_layout(xaxis_title="Lateral Arm (m)", yaxis_title="Weight (kg)", height=450)
st.plotly_chart(fig_lat, use_container_width=True)

# show recent history in sidebar
st.sidebar.header("Recent Inputs")
history_df = load_history(10)
if not history_df.empty:
    st.sidebar.dataframe(history_df[["timestamp", "pilot", "copilot", "takeoff_fuel", "cg_long", "cg_lat", "total_weight"]])
else:
    st.sidebar.write("No history yet.")

st.markdown(
    "For bug reports or feature requests, contact Aryaman Samyal — "
    "[📧 aryamansinghsamyal@gmail.com](mailto:aryamansinghsamyal@gmail.com)."
)

