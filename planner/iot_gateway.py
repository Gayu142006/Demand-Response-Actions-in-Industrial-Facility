"""
IoT / BMS Edge Gateway Ingestion Layer.

Provides protocol adapters for real building management system (BMS) data:
  - BACnet/IP (Building Automation and Control Networks)
  - Modbus TCP/RTU (industrial sensors and PLCs)
  - MQTT (lightweight IoT pub/sub messaging)

Each adapter normalizes incoming telemetry into a common TelemetryPoint
schema that the planner can consume in place of synthetic CSV data.

Status: PROTOCOL ADAPTERS IMPLEMENTED — awaiting real BMS endpoints for
integration testing.
"""
import json
import time
import struct
import socket
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from queue import Queue, Empty

logger = logging.getLogger(__name__)

TELEMETRY_BUFFER_DIR = Path(__file__).resolve().parent.parent / "data" / "telemetry_buffer"
TELEMETRY_BUFFER_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Common telemetry schema
# ---------------------------------------------------------------------------

@dataclass
class TelemetryPoint:
    """Normalized telemetry point that all protocol adapters produce."""
    timestamp: str                    # ISO 8601
    source_protocol: str              # BACNET | MODBUS | MQTT | SYNTHETIC
    device_id: str                    # unique device/sensor identifier
    zone_id: str                      # mapped to facility zone
    metric_type: str                  # TEMPERATURE | LOAD_KW | OCCUPANCY | HUMIDITY | EQUIPMENT_STATUS
    value: float                      # numeric value
    unit: str                         # C, kW, %, etc.
    quality: str = "GOOD"             # GOOD | UNCERTAIN | BAD | STALE
    raw_address: str = ""             # protocol-specific address (e.g., BACnet object, Modbus register)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayHealth:
    """Health status of an edge gateway connection."""
    protocol: str
    endpoint: str
    status: str                       # CONNECTED | DISCONNECTED | ERROR | DEGRADED
    last_successful_read: Optional[str] = None
    error_count: int = 0
    points_received: int = 0
    latency_ms: float = 0.0


@dataclass
class DeviceMapping:
    """Maps a BMS device/register to a facility zone and metric type."""
    device_id: str
    protocol: str
    address: str                      # BACnet object ID, Modbus register, MQTT topic
    zone_id: str
    metric_type: str
    unit: str
    scale_factor: float = 1.0         # multiply raw value by this
    offset: float = 0.0               # add this after scaling


# ---------------------------------------------------------------------------
# Protocol adapters
# ---------------------------------------------------------------------------

class ProtocolAdapter(ABC):
    """Base class for all BMS protocol adapters."""

    def __init__(self, endpoint: str, device_mappings: List[DeviceMapping]):
        self.endpoint = endpoint
        self.device_mappings = {m.address: m for m in device_mappings}
        self.health = GatewayHealth(
            protocol=self.protocol_name(),
            endpoint=endpoint,
            status="DISCONNECTED",
        )
        self._buffer: Queue = Queue(maxsize=10000)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @abstractmethod
    def protocol_name(self) -> str:
        pass

    @abstractmethod
    def _connect(self):
        pass

    @abstractmethod
    def _read_points(self) -> List[TelemetryPoint]:
        pass

    def _disconnect(self):
        self.health.status = "DISCONNECTED"

    def start(self):
        """Start background polling/subscription."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"{self.protocol_name()} gateway started: {self.endpoint}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._disconnect()
        logger.info(f"{self.protocol_name()} gateway stopped: {self.endpoint}")

    def _poll_loop(self):
        while self._running:
            try:
                self._connect()
                points = self._read_points()
                for p in points:
                    self._buffer.put(p, block=False)
                    self.health.points_received += 1
                self.health.last_successful_read = datetime.now(timezone.utc).isoformat()
                self.health.status = "CONNECTED"
                self.health.error_count = 0
            except Exception as e:
                self.health.error_count += 1
                if self.health.error_count > 5:
                    self.health.status = "ERROR"
                else:
                    self.health.status = "DEGRADED"
                logger.warning(f"{self.protocol_name()} read error: {e}")
            time.sleep(15)  # 15-second polling interval matches data resolution

    def drain_buffer(self) -> List[TelemetryPoint]:
        """Drain all buffered telemetry points."""
        points = []
        while True:
            try:
                points.append(self._buffer.get_nowait())
            except Empty:
                break
        return points

    def _apply_mapping(self, address: str, raw_value: float) -> Optional[TelemetryPoint]:
        """Apply device mapping to produce a normalized TelemetryPoint."""
        mapping = self.device_mappings.get(address)
        if not mapping:
            return None
        scaled_value = raw_value * mapping.scale_factor + mapping.offset
        return TelemetryPoint(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_protocol=self.protocol_name(),
            device_id=mapping.device_id,
            zone_id=mapping.zone_id,
            metric_type=mapping.metric_type,
            value=round(scaled_value, 3),
            unit=mapping.unit,
            quality="GOOD",
            raw_address=address,
        )


class BACnetAdapter(ProtocolAdapter):
    """
    BACnet/IP protocol adapter.

    BACnet (Building Automation and Control Networks) is the standard protocol
    for commercial building HVAC, lighting, and access control systems.

    This adapter implements BACnet ReadProperty requests over UDP/IP to poll
    analog-value, analog-input, and binary-value objects from BACnet devices.
    """

    def protocol_name(self) -> str:
        return "BACNET"

    def _connect(self):
        """Establish BACnet/IP communication (UDP socket)."""
        try:
            host, port = self.endpoint.rsplit(":", 1)
            self._host = host
            self._port = int(port) if port else 47808  # BACnet default port
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(5.0)
            self.health.status = "CONNECTED"
        except Exception as e:
            self.health.status = "ERROR"
            raise ConnectionError(f"BACnet connection failed: {e}")

    def _build_read_property_request(self, object_type: int, instance: int, property_id: int = 85) -> bytes:
        """Build a BACnet ReadProperty request PDU.

        Args:
            object_type: BACnet object type (0=analog-input, 1=analog-output, 2=analog-value, etc.)
            instance: Object instance number
            property_id: Property to read (85 = present-value)
        """
        # Simplified BACnet/IP BVLC header + NPDU + APDU
        bvlc_type = 0x81        # BACnet/IP
        bvlc_function = 0x0A    # Original-Unicast-NPDU
        npdu = bytes([0x01, 0x04])  # Version 1, expecting reply

        # Confirmed ReadProperty APDU
        apdu_type = 0x00        # Confirmed request
        service = 0x0C          # ReadProperty
        invoke_id = 0x01

        # Context-tagged object identifier [0]
        obj_id = (object_type << 22) | (instance & 0x3FFFFF)
        obj_bytes = struct.pack(">I", obj_id)

        # Context-tagged property identifier [1]
        prop_bytes = struct.pack(">B", property_id)

        # Build the APDU
        apdu = bytes([apdu_type, 0x05, invoke_id, service])
        apdu += bytes([0x0C]) + obj_bytes  # [0] objectIdentifier
        apdu += bytes([0x19]) + prop_bytes  # [1] propertyIdentifier

        # BVLC header
        length = 4 + len(npdu) + len(apdu)
        bvlc = struct.pack(">BBH", bvlc_type, bvlc_function, length)

        return bvlc + npdu + apdu

    def _parse_bacnet_response(self, data: bytes) -> Optional[float]:
        """Parse a BACnet ReadProperty response to extract a float value."""
        try:
            # Skip BVLC (4 bytes) + NPDU (variable) + APDU header
            # Look for application-tagged real value (tag 0x44 = real, 4 bytes)
            for i in range(len(data) - 4):
                if data[i] == 0x44:  # Application tag: Real
                    value = struct.unpack(">f", data[i + 1:i + 5])[0]
                    return value
            return None
        except Exception:
            return None

    def _read_points(self) -> List[TelemetryPoint]:
        """Read all mapped BACnet objects."""
        points = []
        for address, mapping in self.device_mappings.items():
            try:
                # Parse address format: "AI:0" (analog-input, instance 0)
                parts = address.split(":")
                obj_type_map = {"AI": 0, "AO": 1, "AV": 2, "BI": 3, "BO": 4, "BV": 5}
                obj_type = obj_type_map.get(parts[0], 2)
                instance = int(parts[1]) if len(parts) > 1 else 0

                request = self._build_read_property_request(obj_type, instance)
                self._sock.sendto(request, (self._host, self._port))

                start = time.time()
                data, _ = self._sock.recvfrom(1500)
                self.health.latency_ms = (time.time() - start) * 1000

                value = self._parse_bacnet_response(data)
                if value is not None:
                    point = self._apply_mapping(address, value)
                    if point:
                        points.append(point)
            except socket.timeout:
                logger.debug(f"BACnet timeout reading {address}")
            except Exception as e:
                logger.debug(f"BACnet read error for {address}: {e}")

        return points

    def _disconnect(self):
        if hasattr(self, '_sock'):
            self._sock.close()
        super()._disconnect()


class ModbusAdapter(ProtocolAdapter):
    """
    Modbus TCP/RTU adapter.

    Modbus is the standard protocol for industrial PLCs, power meters, and
    sensor networks. This adapter implements Modbus TCP Function Code 03
    (Read Holding Registers) and FC 04 (Read Input Registers).
    """

    def __init__(self, endpoint: str, device_mappings: List[DeviceMapping]):
        super().__init__(endpoint, device_mappings)
        self._transaction_id = 0

    def protocol_name(self) -> str:
        return "MODBUS"

    def _connect(self):
        """Establish Modbus TCP connection."""
        try:
            host, port = self.endpoint.rsplit(":", 1)
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect((host, int(port)))
            self.health.status = "CONNECTED"
            self._transaction_id = 0
        except Exception as e:
            self.health.status = "ERROR"
            raise ConnectionError(f"Modbus connection failed: {e}")

    def _build_modbus_request(self, unit_id: int, function_code: int,
                               start_register: int, count: int) -> bytes:
        """Build a Modbus TCP request frame."""
        self._transaction_id = (self._transaction_id + 1) % 65536
        # MBAP Header: transaction_id(2) + protocol_id(2) + length(2) + unit_id(1)
        pdu = struct.pack(">BHH", function_code, start_register, count)
        mbap = struct.pack(">HHHB", self._transaction_id, 0, len(pdu) + 1, unit_id)
        return mbap + pdu

    def _parse_modbus_response(self, data: bytes, register_type: str = "INT16") -> Optional[float]:
        """Parse Modbus TCP response to extract register values."""
        try:
            if len(data) < 9:
                return None
            # MBAP header is 7 bytes, then function code (1), byte count (1), data
            byte_count = data[8]
            register_data = data[9:9 + byte_count]

            if register_type == "FLOAT32" and len(register_data) >= 4:
                return struct.unpack(">f", register_data[:4])[0]
            elif register_type == "INT16" and len(register_data) >= 2:
                return struct.unpack(">H", register_data[:2])[0]
            elif register_type == "INT32" and len(register_data) >= 4:
                return struct.unpack(">I", register_data[:4])[0]
            return None
        except Exception:
            return None

    def _read_points(self) -> List[TelemetryPoint]:
        """Read all mapped Modbus registers."""
        points = []
        for address, mapping in self.device_mappings.items():
            try:
                # Parse address format: "1:3:100:INT16"
                # (unit_id:function_code:register:type)
                parts = address.split(":")
                unit_id = int(parts[0]) if len(parts) > 0 else 1
                fc = int(parts[1]) if len(parts) > 1 else 3
                register = int(parts[2]) if len(parts) > 2 else 0
                reg_type = parts[3] if len(parts) > 3 else "INT16"
                count = 2 if reg_type in ("FLOAT32", "INT32") else 1

                request = self._build_modbus_request(unit_id, fc, register, count)

                start = time.time()
                self._sock.send(request)
                data = self._sock.recv(256)
                self.health.latency_ms = (time.time() - start) * 1000

                value = self._parse_modbus_response(data, reg_type)
                if value is not None:
                    point = self._apply_mapping(address, value)
                    if point:
                        points.append(point)
            except Exception as e:
                logger.debug(f"Modbus read error for {address}: {e}")

        return points

    def _disconnect(self):
        if hasattr(self, '_sock'):
            self._sock.close()
        super()._disconnect()


class MQTTAdapter(ProtocolAdapter):
    """
    MQTT protocol adapter.

    MQTT is a lightweight pub/sub messaging protocol widely used in IoT.
    This adapter subscribes to configurable topics and converts incoming
    JSON payloads into normalized TelemetryPoints.

    Expected payload format:
        {"value": 23.5, "unit": "C", "quality": "GOOD"}
    or simply a raw numeric value.
    """

    def protocol_name(self) -> str:
        return "MQTT"

    def _connect(self):
        """Establish MQTT connection (lightweight implementation)."""
        try:
            host_port = self.endpoint.replace("mqtt://", "").replace("tcp://", "")
            parts = host_port.rsplit(":", 1)
            self._host = parts[0]
            self._port = int(parts[1]) if len(parts) > 1 else 1883
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect((self._host, self._port))

            # MQTT CONNECT packet
            client_id = b"oadrp_gateway"
            connect_flags = 0x02  # Clean session
            keepalive = 60

            # Variable header
            var_header = b"\x00\x04MQTT"  # Protocol name
            var_header += bytes([0x04])     # Protocol level (MQTT 3.1.1)
            var_header += bytes([connect_flags])
            var_header += struct.pack(">H", keepalive)

            # Payload: client ID
            payload = struct.pack(">H", len(client_id)) + client_id

            # Fixed header
            remaining = len(var_header) + len(payload)
            fixed = bytes([0x10]) + self._encode_remaining_length(remaining)

            self._sock.send(fixed + var_header + payload)
            connack = self._sock.recv(4)

            if len(connack) >= 4 and connack[3] == 0x00:
                self.health.status = "CONNECTED"
                # Subscribe to all mapped topics
                self._subscribe_all()
            else:
                raise ConnectionError("MQTT CONNACK rejected")

        except Exception as e:
            self.health.status = "ERROR"
            raise ConnectionError(f"MQTT connection failed: {e}")

    def _encode_remaining_length(self, length: int) -> bytes:
        """Encode MQTT remaining length field."""
        result = bytearray()
        while True:
            byte = length % 128
            length = length // 128
            if length > 0:
                byte |= 0x80
            result.append(byte)
            if length == 0:
                break
        return bytes(result)

    def _subscribe_all(self):
        """Subscribe to all configured MQTT topics."""
        for topic in self.device_mappings.keys():
            topic_bytes = topic.encode("utf-8")
            # SUBSCRIBE packet
            packet_id = 1
            payload = struct.pack(">H", len(topic_bytes)) + topic_bytes + bytes([0x00])
            var_header = struct.pack(">H", packet_id)
            remaining = len(var_header) + len(payload)
            fixed = bytes([0x82]) + self._encode_remaining_length(remaining)
            self._sock.send(fixed + var_header + payload)

    def _read_points(self) -> List[TelemetryPoint]:
        """Read incoming MQTT PUBLISH messages."""
        points = []
        self._sock.settimeout(2.0)
        try:
            data = self._sock.recv(4096)
            if data and (data[0] & 0xF0) == 0x30:  # PUBLISH
                # Parse PUBLISH packet
                idx = 1
                remaining, bytes_used = self._decode_remaining_length(data[idx:])
                idx += bytes_used

                topic_len = struct.unpack(">H", data[idx:idx + 2])[0]
                idx += 2
                topic = data[idx:idx + topic_len].decode("utf-8")
                idx += topic_len
                payload = data[idx:idx + remaining - 2 - topic_len]

                # Parse payload
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        value = float(parsed.get("value", 0))
                    else:
                        value = float(parsed)
                except (json.JSONDecodeError, ValueError):
                    try:
                        value = float(payload)
                    except ValueError:
                        return points

                point = self._apply_mapping(topic, value)
                if point:
                    points.append(point)
        except socket.timeout:
            pass
        return points

    def _decode_remaining_length(self, data: bytes) -> tuple:
        """Decode MQTT remaining length field."""
        multiplier = 1
        value = 0
        idx = 0
        while idx < len(data):
            byte = data[idx]
            value += (byte & 0x7F) * multiplier
            multiplier *= 128
            idx += 1
            if (byte & 0x80) == 0:
                break
        return value, idx

    def _disconnect(self):
        if hasattr(self, '_sock'):
            try:
                self._sock.send(bytes([0xE0, 0x00]))  # DISCONNECT
            except Exception:
                pass
            self._sock.close()
        super()._disconnect()


# ---------------------------------------------------------------------------
# Edge Gateway Manager
# ---------------------------------------------------------------------------

class EdgeGateway:
    """
    Manages multiple protocol adapters and provides a unified telemetry
    stream to the planner.

    Falls back to synthetic data when no real BMS endpoints are reachable.
    """

    def __init__(self):
        self._adapters: Dict[str, ProtocolAdapter] = {}
        self._telemetry_log: List[TelemetryPoint] = []
        self._callbacks: List[Callable[[TelemetryPoint], None]] = []
        self._fallback_to_synthetic = True

    def add_adapter(self, name: str, adapter: ProtocolAdapter):
        self._adapters[name] = adapter

    def add_callback(self, callback: Callable[[TelemetryPoint], None]):
        """Register a callback for new telemetry points."""
        self._callbacks.append(callback)

    def start_all(self):
        for name, adapter in self._adapters.items():
            try:
                adapter.start()
                logger.info(f"Started adapter: {name}")
            except Exception as e:
                logger.error(f"Failed to start adapter {name}: {e}")

    def stop_all(self):
        for name, adapter in self._adapters.items():
            adapter.stop()

    def collect_telemetry(self) -> List[TelemetryPoint]:
        """Collect all new telemetry points from all adapters."""
        all_points = []
        for name, adapter in self._adapters.items():
            points = adapter.drain_buffer()
            all_points.extend(points)

        # Notify callbacks
        for point in all_points:
            for cb in self._callbacks:
                try:
                    cb(point)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        self._telemetry_log.extend(all_points)

        # Buffer to disk for persistence
        if all_points:
            self._persist_points(all_points)

        return all_points

    def _persist_points(self, points: List[TelemetryPoint]):
        """Persist telemetry points to a local buffer file."""
        now = datetime.now(timezone.utc)
        filename = f"telemetry_{now.strftime('%Y%m%d_%H')}.jsonl"
        path = TELEMETRY_BUFFER_DIR / filename
        with open(path, "a") as f:
            for p in points:
                f.write(json.dumps(asdict(p)) + "\n")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all adapters."""
        return {
            name: {
                "protocol": adapter.health.protocol,
                "endpoint": adapter.health.endpoint,
                "status": adapter.health.status,
                "last_successful_read": adapter.health.last_successful_read,
                "error_count": adapter.health.error_count,
                "points_received": adapter.health.points_received,
                "latency_ms": round(adapter.health.latency_ms, 1),
            }
            for name, adapter in self._adapters.items()
        }

    def build_facility_snapshot(self, points: List[TelemetryPoint]) -> Dict[str, Any]:
        """
        Convert a batch of telemetry points into the facility snapshot format
        expected by planner_engine.run_planner().
        """
        zone_temperatures = {}
        equipment_status = {}
        total_load_kw = 0.0
        production_load_kw = 0.0

        for p in points:
            if p.metric_type == "TEMPERATURE":
                zone_temperatures[p.zone_id] = p.value
            elif p.metric_type == "LOAD_KW":
                total_load_kw += p.value
                if "PRODUCTION" in p.zone_id.upper():
                    production_load_kw += p.value
            elif p.metric_type == "EQUIPMENT_STATUS":
                status = "AVAILABLE" if p.value > 0 else "MAINTENANCE"
                equipment_status[p.device_id] = status

        return {
            "zone_temperatures": zone_temperatures,
            "equipment_status": equipment_status,
            "total_load_kw": total_load_kw,
            "production_load_kw": production_load_kw,
            "source": "LIVE_TELEMETRY",
        }


# ---------------------------------------------------------------------------
# Example configuration for the synthetic facility
# ---------------------------------------------------------------------------

def create_example_bacnet_mappings() -> List[DeviceMapping]:
    """Example BACnet device mappings for the synthetic facility."""
    return [
        DeviceMapping("TEMP_OFFICE_A", "BACNET", "AV:1", "OFFICE_A", "TEMPERATURE", "C"),
        DeviceMapping("TEMP_WAREHOUSE", "BACNET", "AV:2", "WAREHOUSE", "TEMPERATURE", "C"),
        DeviceMapping("TEMP_PRODUCTION", "BACNET", "AV:3", "PRODUCTION", "TEMPERATURE", "C"),
        DeviceMapping("TEMP_CONTROL_ROOM", "BACNET", "AV:4", "CONTROL_ROOM", "TEMPERATURE", "C"),
        DeviceMapping("TEMP_UTILITY", "BACNET", "AV:5", "UTILITY", "TEMPERATURE", "C"),
        DeviceMapping("HVAC_OFFICE_LOAD", "BACNET", "AI:10", "OFFICE_A", "LOAD_KW", "kW"),
        DeviceMapping("HVAC_WAREHOUSE_LOAD", "BACNET", "AI:11", "WAREHOUSE", "LOAD_KW", "kW"),
    ]


def create_example_modbus_mappings() -> List[DeviceMapping]:
    """Example Modbus register mappings for power meters and PLCs."""
    return [
        DeviceMapping("POWER_METER_1", "MODBUS", "1:3:100:FLOAT32", "PRODUCTION", "LOAD_KW", "kW",
                      scale_factor=0.001),
        DeviceMapping("POWER_METER_2", "MODBUS", "1:3:102:FLOAT32", "UTILITY", "LOAD_KW", "kW",
                      scale_factor=0.001),
        DeviceMapping("COMPRESSOR_A_STATUS", "MODBUS", "2:3:200:INT16", "PRODUCTION", "EQUIPMENT_STATUS", "",
                      scale_factor=1.0),
    ]


def create_example_mqtt_mappings() -> List[DeviceMapping]:
    """Example MQTT topic mappings for IoT sensors."""
    return [
        DeviceMapping("IOT_TEMP_OFFICE", "MQTT", "facility/office_a/temperature", "OFFICE_A", "TEMPERATURE", "C"),
        DeviceMapping("IOT_OCC_OFFICE", "MQTT", "facility/office_a/occupancy", "OFFICE_A", "OCCUPANCY", "%"),
        DeviceMapping("IOT_TEMP_WAREHOUSE", "MQTT", "facility/warehouse/temperature", "WAREHOUSE", "TEMPERATURE", "C"),
        DeviceMapping("IOT_HUMIDITY_OFFICE", "MQTT", "facility/office_a/humidity", "OFFICE_A", "HUMIDITY", "%"),
    ]
