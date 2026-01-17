import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta, date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
MASTER_CSV = 'master_edge_sensor_data.csv'
V_REF = 5.0        
ADC_MAX = 1023     

# Defined Safety Limits
SAFE_LIMITS = {
    "Temperature": (2.0, 4.0),
    "Moisture": (1.2, 3.0),
    "Light": (0.0, 5.0)
}

# --- 1. Page Config ---
st.set_page_config(
    page_title="SensorEdge", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Helper: Data Cleaning (CRITICAL FIX) ---
def clean_dataframe(df):
    """
    Standardizes column names to prevent KeyError.
    """
    # 1. Clean Headers
    df.columns = df.columns.str.strip().str.title()
    
    # 2. Rename Map (Handles variations)
    rename_map = {
        'Time': 'Timestamp', 'Date': 'Timestamp', 
        'Sensor': 'Sensor_Name', 'Node': 'Sensor_Name',
        'Voltage': 'Voltage_V', 'Volts': 'Voltage_V', 
        'Adc': 'ADC_Value', 'Adc_Value': 'ADC_Value', 'Adc Value': 'ADC_Value'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 3. Drop Duplicate Columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 4. Ensure Required Columns Exist
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.round('1s')
    
    if 'Sensor_Name' in df.columns:
        df['Sensor_Name'] = df['Sensor_Name'].apply(normalize_sensor_names)
        
    # 5. Fix Numeric Types (Prevent crash if missing)
    if 'Voltage_V' not in df.columns: df['Voltage_V'] = 0.0
    if 'ADC_Value' not in df.columns: df['ADC_Value'] = 0
    
    df['Voltage_V'] = pd.to_numeric(df['Voltage_V'], errors='coerce').fillna(0.0)
    df['ADC_Value'] = pd.to_numeric(df['ADC_Value'], errors='coerce').fillna(0).astype(int)
    
    if 'Status' not in df.columns:
        df['Status'] = 'New'
        
    return df

def normalize_sensor_names(name):
    n = str(name).strip().lower()
    if 'temp' in n or 'node_1' in n: return 'Temperature'
    if 'moist' in n or 'node_2' in n: return 'Moisture'
    if 'light' in n or 'node_3' in n: return 'Light'
    return n.title()

# --- 3. Logic: Generation ---
def generate_sensor_data(sensor_name, interval_seconds, v_min, v_max, duration_hours, start_time):
    total_seconds = duration_hours * 3600
    timestamps = [start_time + timedelta(seconds=i) for i in range(0, total_seconds, interval_seconds)]
    
    data = []
    for ts in timestamps:
        voltage = np.random.uniform(v_min, v_max)
        adc_value = int((voltage / V_REF) * ADC_MAX)
        
        data.append({
            "Timestamp": ts,
            "Sensor_Name": sensor_name,
            "Voltage_V": round(voltage, 2),
            "ADC_Value": adc_value,
            "Status": "Historical"
        })
    return pd.DataFrame(data)

def generate_full_dataset(start_date_obj):
    start_time = datetime.combine(start_date_obj, datetime.min.time())
    duration = 24 
    
    df_temp = generate_sensor_data("Temperature", 2, 2.0, 4.0, duration, start_time)
    df_moist = generate_sensor_data("Moisture", 14400, 1.2, 3.0, duration, start_time)
    df_light = generate_sensor_data("Light", 5, 0.0, 5.0, duration, start_time)
    
    full_df = pd.concat([df_temp, df_moist, df_light])
    full_df = full_df.sort_values(by="Timestamp").reset_index(drop=True)
    return full_df

# --- 4. Logic: Merge ---
def merge_logic_exact(master_df, new_df):
    # 1. Clean Incoming Data
    new_df = clean_dataframe(new_df)
    
    # 2. Mark Overlaps
    if not master_df.empty:
        # Ensure Master is clean too
        master_df = clean_dataframe(master_df) 
        master_df['Status'] = 'Historical'
        
        m_idx = master_df.set_index(['Timestamp', 'Sensor_Name']).index
        n_idx = new_df.set_index(['Timestamp', 'Sensor_Name']).index
        master_df.loc[m_idx.isin(n_idx), 'Status'] = 'Overlap'

    # 3. Core Merge Logic
    combined = pd.concat([master_df, new_df])
    
    # "keep='first' to preserve Master data"
    final_df = combined.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
    final_df = final_df.sort_values(by='Timestamp').reset_index(drop=True)
    
    # Stats
    stats = {
        "old": len(master_df),
        "new_file": len(new_df),
        "final": len(final_df),
        "added": len(final_df) - len(master_df)
    }
    stats["ignored"] = stats["new_file"] - stats["added"]
    
    return final_df, stats

# --- 5. Logic: Compression ---
def analyze_compression(df):
    results = []
    total_raw_bytes = 0
    total_comp_bytes = 0
    
    for sensor, sub_raw in df.groupby('Sensor_Name'):
        sub = sub_raw.sort_values('Timestamp').copy()
        adc_values = sub['ADC_Value'].values
        
        if len(adc_values) == 0: continue
            
        raw_size = len(adc_values) * 2
        comp_size = 2 
        
        if len(adc_values) > 1:
            deltas = np.diff(adc_values)
            small_deltas = (deltas >= -127) & (deltas <= 127)
            num_small = np.sum(small_deltas)
            num_large = len(deltas) - num_small
            comp_size += (num_small * 1) + (num_large * 2)
            
        saved = raw_size - comp_size
        pct = (saved / raw_size) * 100 if raw_size > 0 else 0
        
        total_raw_bytes += raw_size
        total_comp_bytes += comp_size
        
        results.append({
            "Sensor": sensor,
            "Samples": len(adc_values),
            "Raw Size (KB)": round(raw_size / 1024, 2),
            "Compressed (KB)": round(comp_size / 1024, 2),
            "Saved (%)": round(pct, 1)
        })
        
    overall_saved = total_raw_bytes - total_comp_bytes
    overall_pct = (overall_saved / total_raw_bytes * 100) if total_raw_bytes > 0 else 0
    
    summary = {
        "Total Raw": round(total_raw_bytes / 1024, 2),
        "Total Compressed": round(total_comp_bytes / 1024, 2),
        "Total Saved KB": round(overall_saved / 1024, 2),
        "Total Saved %": round(overall_pct, 1)
    }
    
    return pd.DataFrame(results), summary

# --- 6. Logic: Error Detection ---
def check_system_health(df):
    alerts = []
    for sensor, sub in df.groupby('Sensor_Name'):
        v_min, v_max = SAFE_LIMITS.get(sensor, (0, 5))
        
        high_mask = sub['Voltage_V'] > v_max
        if high_mask.any():
            count = high_mask.sum()
            alerts.append({
                "Severity": "Critical",
                "Sensor": sensor,
                "Issue": f"Voltage High (> {v_max}V)",
                "Count": f"{count} records",
                "Status": "Over Limit"
            })
            
        low_mask = sub['Voltage_V'] < v_min
        if low_mask.any():
            count = low_mask.sum()
            alerts.append({
                "Severity": "Warning",
                "Sensor": sensor,
                "Issue": f"Voltage Low (< {v_min}V)",
                "Count": f"{count} records",
                "Status": "Under Limit"
            })
            
        if len(sub) > 100 and sub['Voltage_V'].std() < 0.01:
             alerts.append({
                "Severity": "Critical",
                "Sensor": sensor,
                "Issue": "Sensor Frozen",
                "Count": "All",
                "Status": "Stuck"
            })
    return pd.DataFrame(alerts)

# --- 7. Logic: Analytics ---
def get_analytics(df):
    # CRITICAL: Re-verify columns here just in case
    if df.empty or 'ADC_Value' not in df.columns: 
        # Attempt to clean one last time if missing
        df = clean_dataframe(df.copy())
        if 'ADC_Value' not in df.columns: return {}

    results = {}
    
    for sensor, sub_raw in df.groupby('Sensor_Name'):
        sub = sub_raw.copy()
        
        if len(sub) > 0:
            exp = (sub['Voltage_V'] / V_REF * ADC_MAX).astype(int)
            hw = (np.abs(sub['ADC_Value'] - exp) <= 1).mean() * 100
        else:
            hw = 0.0
        results[sensor] = {'count': len(sub), 'hw': hw, 'ai': 0.0}
    
    # AI Accuracy
    if df['Sensor_Name'].nunique() >= 2:
        try:
            clean = df.dropna(subset=['Voltage_V', 'Sensor_Name'])
            X = clean[['Voltage_V']]
            y = clean['Sensor_Name']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            for s in results:
                if s in report: 
                    results[s]['ai'] = report[s]['f1-score'] * 100
        except: pass
        
    return results

# --- 8. Main App ---

# Initial Load Logic: Clean data immediately on load
if 'master_df' not in st.session_state:
    if os.path.exists(MASTER_CSV):
        try:
            raw_df = pd.read_csv(MASTER_CSV)
            st.session_state.master_df = clean_dataframe(raw_df)
        except:
            st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value'])
    else:
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value'])

master_df = st.session_state.master_df

# Sidebar
with st.sidebar:
    st.title("Control Panel")
    st.markdown("---")
    
    st.subheader("1. Initialize Master Data")
    
    # Option A: Upload Master
    master_upload = st.file_uploader("Upload Existing Master CSV (Optional)", type=['csv'], key="master_uploader")
    if master_upload:
        if st.button("Load Master from File"):
            try:
                loaded_df = pd.read_csv(master_upload)
                # CLEAN IMMEDIATELY
                loaded_df = clean_dataframe(loaded_df)
                
                if not loaded_df.empty and 'Timestamp' in loaded_df.columns:
                    loaded_df.to_csv(MASTER_CSV, index=False)
                    st.session_state.master_df = loaded_df
                    st.success("Master File Loaded Successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid CSV structure.")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # Option B: Generate Random
    st.write("OR")
    if st.button("Generate Random Master & Reset"):
        with st.spinner("Generating..."):
            start_date = datetime.now().date()
            new_data = generate_full_dataset(start_date)
            # Ensure generated data is clean (it should be, but good practice)
            new_data = clean_dataframe(new_data)
            new_data.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = new_data
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    st.subheader("2. Merge New Data")
    uploaded = st.file_uploader("Upload Sensor Data CSV", type=['csv'])
    
    if uploaded:
        if st.button("Run Merge Script"):
            try:
                new_raw = pd.read_csv(uploaded)
                final_df, stats = merge_logic_exact(master_df, new_raw)
                
                final_df.to_csv(MASTER_CSV, index=False)
                st.session_state.master_df = final_df
                
                msg = f"Merge Complete! Added: {stats['added']} | Ignored: {stats['ignored']}"
                st.toast(msg, icon="✅")
                st.success(msg)
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Merge Failed: {e}")

    st.markdown("---")
    if st.button("Factory Reset"):
        if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value'])
        st.rerun()

# --- Main Dashboard ---
st.title("SensorEdge")

if not master_df.empty:
    # Ensure master_df is clean before using
    if 'ADC_Value' not in master_df.columns:
        master_df = clean_dataframe(master_df)
        st.session_state.master_df = master_df

    k1, k2, k3, k4 = st.columns(4)
    analytics = get_analytics(master_df)
    
    avg_hw = np.mean([v['hw'] for v in analytics.values()]) if analytics else 0.0
    avg_ai = np.mean([v['ai'] for v in analytics.values()]) if analytics else 0.0
    
    k1.metric("Total Records", f"{len(master_df):,}")
    k2.metric("Active Sensors", master_df['Sensor_Name'].nunique())
    k3.metric("Hardware Accuracy", f"{avg_hw:.1f}%")
    k4.metric("AI Accuracy", f"{avg_ai:.1f}%")
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Rows Value/Counts", "Data Inspector", "System Requirements", "Compression Lab"])
    
    with tab1:
        if analytics:
            rows = [{"Sensor": s, "Count": f"{m['count']:,}", "HW Accuracy": m['hw']/100, "AI Accuracy": m['ai']/100} for s, m in analytics.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                column_config={
                    "HW Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                    "AI Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1)
                })
        else:
            st.info("No valid data for analytics.")
            
    with tab2:
        st.caption(f"Showing sample of {len(master_df):,} rows.")
        
        def highlight_status(val):
            if val == 'New': return 'background-color: #d1e7dd; color: #0f5132'
            if val == 'Overlap': return 'background-color: #f8d7da; color: #842029'
            return ''
        
        view = master_df.head(1000).copy()
        try:
            st.dataframe(view.style.map(highlight_status, subset=['Status']), use_container_width=True)
        except:
            st.dataframe(view, use_container_width=True)
            
        st.download_button("Download Merge_CSV", master_df.to_csv(index=False).encode('utf-8'), "Merge.csv", "text/csv")
        
    with tab3: # System Health
        st.markdown("### Anomaly Detection Logic")
        health_df = check_system_health(master_df)
        
        if not health_df.empty:
            c1, c2 = st.columns(2)
            c1.error(f"{len(health_df)} Issues Detected")
            c2.warning("Action Required: Check sensor calibration.")
            st.dataframe(health_df, use_container_width=True, hide_index=True)
        else:
            st.success("All Systems Nominal.")
            st.markdown("""
            * **Temperature:** 2.0V - 4.0V
            * **Moisture:** 1.2V - 3.0V
            * **Light:** 0.0V - 5.0V
            """)

    with tab4: # Compression Lab
        st.markdown("### Frame Difference Compression Analysis")
        st.info("Calculates data savings by storing the **difference** (Delta) instead of full values.")
        
        if st.button("Run Compression Analysis"):
            with st.spinner("Compressing data..."):
                comp_df, comp_summary = analyze_compression(master_df)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Raw Size", f"{comp_summary['Total Raw']} KB")
                m2.metric("Compressed", f"{comp_summary['Total Compressed']} KB")
                m3.metric("Space Saved", f"{comp_summary['Total Saved KB']} KB", delta=f"{comp_summary['Total Saved %']}%")
                m4.metric("Method", "Delta Encoding")
                
                st.dataframe(comp_df, use_container_width=True)
                        
else:
    st.info("System Offline. Use sidebar to Initialize Data.")
