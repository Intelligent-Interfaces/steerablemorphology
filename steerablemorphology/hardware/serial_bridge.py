import json
import time
from typing import Dict, Any, Optional

class MicrofluidicSerialBridge:
    """
    Serial communications bridge translating high-level parameter vectors
    into microfluidic pump step rates and LED PWM power levels.
    """
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, mock: bool = True):
        self.port = port
        self.baudrate = baudrate
        self.mock = mock
        self.last_dispatched: Dict[str, Any] = {}
        self.is_connected = not mock

    def connect(self) -> bool:
        if self.mock:
            self.is_connected = True
            return True
        try:
            import serial
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.is_connected = True
            return True
        except Exception as e:
            print(f"[SerialBridge] Connection failed: {e}")
            self.is_connected = False
            return False

    def send_actuation_packet(self, q_syringe: float, p_laser: float, q_dial: float) -> Dict[str, Any]:
        """
        Sends formatted JSON actuation packet to microcontroller.
        
        Args:
            q_syringe: Flow rate in uL/min (0.0 to 2.0).
            p_laser: Laser power in mW (0 to 50).
            q_dial: Dialysis flow rate in uL/min (0.0 to 5.0).
        """
        packet = {
            "type": "ACTUATE",
            "timestamp": time.time(),
            "q_syringe_ul_min": round(q_syringe, 3),
            "p_laser_mw": round(p_laser, 1),
            "q_dial_ul_min": round(q_dial, 3)
        }
        self.last_dispatched = packet
        if not self.mock and self.is_connected:
            msg = (json.dumps(packet) + "\n").encode("utf-8")
            self.serial_conn.write(msg)
        return packet
