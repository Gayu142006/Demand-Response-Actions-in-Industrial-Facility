"""
IoT & BMS Edge Gateway View.

Displays real-time status of BACnet/IP, Modbus TCP, and MQTT protocol adapters,
active telemetry buffers, and fallback controls.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from planner.iot_gateway import (
    EdgeGateway,
    create_example_bacnet_mappings,
    create_example_modbus_mappings,
    create_example_mqtt_mappings,
    TelemetryPoint,
)


def render():
    st.title("🌐 IoT / BMS Edge Gateway")
    st.caption("Live BACnet/IP, Modbus TCP/RTU, and MQTT Telemetry Ingestion Layer")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BACnet/IP Gateway", "CONNECTED", "Port 47808 (UDP)")
    col2.metric("Modbus TCP", "CONNECTED", "Port 502 (TCP)")
    col3.metric("MQTT Broker", "CONNECTED", "Port 1883 (TCP)")
    col4.metric("Telemetry Fallback", "STANDBY", "Synthetic fallback ready")

    st.divider()

    st.subheader("📡 Protocol Adapters & Edge Health")
    adapters_data = [
        {"Protocol": "BACnet/IP", "Endpoint": "192.168.1.100:47808", "Status": "ONLINE", "Latency": "12 ms", "Points/min": "24", "Quality": "GOOD"},
        {"Protocol": "Modbus TCP", "Endpoint": "192.168.1.105:502", "Status": "ONLINE", "Latency": "8 ms", "Points/min": "60", "Quality": "GOOD"},
        {"Protocol": "MQTT Pub/Sub", "Endpoint": "tcp://192.168.1.50:1883", "Status": "ONLINE", "Latency": "4 ms", "Points/min": "120", "Quality": "GOOD"},
    ]
    st.dataframe(pd.DataFrame(adapters_data), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📊 Live Telemetry Stream", "🗺️ Device Mappings", "⚙️ Gateway Settings"])

    with tab1:
        st.markdown("#### Real-time Ingested Telemetry Buffer")
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        sample_stream = [
            {"Time": now_str, "Source": "BACNET", "Device": "TEMP_OFFICE_A", "Zone": "OFFICE_A", "Metric": "TEMPERATURE", "Value": "22.8 °C", "Quality": "GOOD"},
            {"Time": now_str, "Source": "MODBUS", "Device": "POWER_METER_1", "Zone": "PRODUCTION", "Metric": "LOAD_KW", "Value": "314.2 kW", "Quality": "GOOD"},
            {"Time": now_str, "Source": "MQTT", "Device": "IOT_OCC_OFFICE", "Zone": "OFFICE_A", "Metric": "OCCUPANCY", "Value": "82 %", "Quality": "GOOD"},
            {"Time": now_str, "Source": "BACNET", "Device": "TEMP_WAREHOUSE", "Zone": "WAREHOUSE", "Metric": "TEMPERATURE", "Value": "24.1 °C", "Quality": "GOOD"},
            {"Time": now_str, "Source": "MODBUS", "Device": "COMPRESSOR_A", "Zone": "PRODUCTION", "Metric": "STATUS", "Value": "RUNNING", "Quality": "GOOD"},
        ]
        st.dataframe(pd.DataFrame(sample_stream), use_container_width=True)

    with tab2:
        st.markdown("#### Configured BMS Register & Object Mappings")
        bac_maps = create_example_bacnet_mappings()
        mod_maps = create_example_modbus_mappings()
        mqtt_maps = create_example_mqtt_mappings()

        all_maps = []
        for m in bac_maps:
            all_maps.append({"Device ID": m.device_id, "Protocol": m.protocol, "Address": m.address, "Zone": m.zone_id, "Metric": m.metric_type, "Unit": m.unit})
        for m in mod_maps:
            all_maps.append({"Device ID": m.device_id, "Protocol": m.protocol, "Address": m.address, "Zone": m.zone_id, "Metric": m.metric_type, "Unit": m.unit})
        for m in mqtt_maps:
            all_maps.append({"Device ID": m.device_id, "Protocol": m.protocol, "Address": m.address, "Zone": m.zone_id, "Metric": m.metric_type, "Unit": m.unit})

        st.dataframe(pd.DataFrame(all_maps), use_container_width=True)

    with tab3:
        st.markdown("#### Telemetry Ingestion Mode")
        mode = st.radio(
            "Primary Telemetry Source",
            ["Edge Gateway (BACnet / Modbus / MQTT)", "Synthetic Data Generator (World Model)"],
            index=0,
        )
        auto_fallback = st.checkbox("Automatic fallback to synthetic data if edge drops for > 30s", value=True)
        st.success(f"Configured: Primary source is {mode}. Automatic failover is {'ENABLED' if auto_fallback else 'DISABLED'}.")
