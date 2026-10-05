"""
Unit tests for the IoT/BMS Edge Gateway Ingestion layer (BACnet, Modbus, MQTT).
"""
import pytest
import struct
from planner.iot_gateway import (
    TelemetryPoint,
    DeviceMapping,
    BACnetAdapter,
    ModbusAdapter,
    MQTTAdapter,
    EdgeGateway,
    create_example_bacnet_mappings,
    create_example_modbus_mappings,
    create_example_mqtt_mappings,
)


def test_bacnet_request_builder():
    mappings = create_example_bacnet_mappings()
    adapter = BACnetAdapter(endpoint="127.0.0.1:47808", device_mappings=mappings)
    assert adapter.protocol_name() == "BACNET"
    assert adapter.health.status == "DISCONNECTED"

    # Test packet builder for Analog Value, instance 1
    req = adapter._build_read_property_request(object_type=2, instance=1, property_id=85)
    assert len(req) > 10
    # BVLC Type should be 0x81 (BACnet/IP)
    assert req[0] == 0x81
    assert req[1] == 0x0A


def test_bacnet_response_parser():
    adapter = BACnetAdapter(endpoint="127.0.0.1:47808", device_mappings=[])
    # Build simulated response containing Real tag 0x44 followed by IEEE 754 float 23.5
    raw_payload = b"\x81\x0A\x00\x11\x01\x00\x10\x03\x01\x0C\x44" + struct.pack(">f", 23.5)
    val = adapter._parse_bacnet_response(raw_payload)
    assert val is not None
    assert round(val, 1) == 23.5


def test_modbus_request_and_response():
    mappings = create_example_modbus_mappings()
    adapter = ModbusAdapter(endpoint="127.0.0.1:502", device_mappings=mappings)
    assert adapter.protocol_name() == "MODBUS"

    # Test request builder
    req = adapter._build_modbus_request(unit_id=1, function_code=3, start_register=100, count=2)
    assert len(req) == 12  # 7 MBAP + 5 PDU

    # Test FLOAT32 response parsing
    # MBAP (7 bytes) + function code 3 (1 byte) + byte count 4 (1 byte) + float 450.75
    sim_data = b"\x00\x01\x00\x00\x00\x07\x01\x03\x04" + struct.pack(">f", 450.75)
    val = adapter._parse_modbus_response(sim_data, register_type="FLOAT32")
    assert val is not None
    assert round(val, 2) == 450.75

    # Test INT16 response parsing
    sim_int16 = b"\x00\x02\x00\x00\x00\x05\x02\x03\x02" + struct.pack(">H", 1)
    val_int = adapter._parse_modbus_response(sim_int16, register_type="INT16")
    assert val_int == 1


def test_mqtt_remaining_length_encoding_decoding():
    adapter = MQTTAdapter(endpoint="tcp://127.0.0.1:1883", device_mappings=[])
    assert adapter.protocol_name() == "MQTT"

    for length in [0, 64, 127, 128, 321, 16383]:
        encoded = adapter._encode_remaining_length(length)
        decoded, _ = adapter._decode_remaining_length(encoded)
        assert decoded == length


def test_edge_gateway_snapshot_generation():
    gw = EdgeGateway()
    points = [
        TelemetryPoint(
            timestamp="2026-10-05T12:00:00Z",
            source_protocol="BACNET",
            device_id="TEMP_OFFICE_A",
            zone_id="OFFICE_A",
            metric_type="TEMPERATURE",
            value=23.4,
            unit="C",
        ),
        TelemetryPoint(
            timestamp="2026-10-05T12:00:00Z",
            source_protocol="MODBUS",
            device_id="POWER_METER_1",
            zone_id="PRODUCTION",
            metric_type="LOAD_KW",
            value=310.5,
            unit="kW",
        ),
        TelemetryPoint(
            timestamp="2026-10-05T12:00:00Z",
            source_protocol="MODBUS",
            device_id="COMPRESSOR_A",
            zone_id="PRODUCTION",
            metric_type="EQUIPMENT_STATUS",
            value=1.0,
            unit="",
        ),
    ]

    snapshot = gw.build_facility_snapshot(points)
    assert snapshot["source"] == "LIVE_TELEMETRY"
    assert snapshot["zone_temperatures"]["OFFICE_A"] == 23.4
    assert snapshot["total_load_kw"] == 310.5
    assert snapshot["production_load_kw"] == 310.5
    assert snapshot["equipment_status"]["COMPRESSOR_A"] == "AVAILABLE"
