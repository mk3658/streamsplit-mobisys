"""
Resource monitoring for edge devices (paper Sec. 4.2.2, Table 8 state space).

Tracks the two system-state components of the Control Plane's MDP state
s_t = [U_t, R_cpu, B_net]: CPU utilization R_cpu (real, via psutil) and
uplink bandwidth B_net (an exponential moving average over recent
transmission times). Embedding uncertainty U_t is provided separately by
edge.distributional_memory.DistributionalMemory.entropy().

Battery/thermal readings are kept as auxiliary diagnostics (genuine psutil
signals) but are not part of the RL-facing state, since Table 8 specifies
exactly three state dimensions.
"""

import time
from collections import deque
from threading import Lock, Thread
from typing import Dict

import psutil


class ResourceMonitor:
    """Monitor CPU load and uplink bandwidth on the edge device."""

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config['edge']['resource_monitor']
        self.poll_interval = self.config['poll_interval_ms'] / 1000.0
        self.bandwidth_ema_alpha = self.config.get('bandwidth_ema_alpha', 0.3)

        self.state = {
            'cpu_util': 0.0,
            'mem_usage': 0.0,
            'battery_level': 1.0,
            'thermal_throttling': 0.0,
        }
        self._bandwidth_mbps = self.config.get('bandwidth_init_mbps', 10.0)
        self._transmission_log = deque(maxlen=100)

        self.lock = Lock()
        self.running = False
        self.monitor_thread = None

    def start(self):
        """Start CPU/memory monitoring in a background thread."""
        if not self.running:
            self.running = True
            self.monitor_thread = Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitor_thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """Background CPU/memory/battery/thermal polling loop."""
        while self.running:
            cpu_util = psutil.cpu_percent(interval=None) / 100.0
            mem_usage = psutil.virtual_memory().percent / 100.0

            battery_level = 1.0
            try:
                battery = psutil.sensors_battery()
                if battery:
                    battery_level = battery.percent / 100.0
            except Exception:
                pass

            thermal_throttling = 0.0
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for entries in temps.values():
                        if any(entry.current > 80 for entry in entries):
                            thermal_throttling = 1.0
                            break
            except Exception:
                pass

            with self.lock:
                self.state = {
                    'cpu_util': cpu_util,
                    'mem_usage': mem_usage,
                    'battery_level': battery_level,
                    'thermal_throttling': thermal_throttling,
                }

            time.sleep(self.poll_interval)

    def record_transmission(self, num_bytes: int, duration_s: float):
        """
        Record a completed transmission to update the bandwidth EMA B_net
        (Sec. 4.2.2: "computed via exponential moving average over recent
        transmission times").

        Args:
            num_bytes: Payload size in bytes
            duration_s: Wall-clock transmission duration in seconds
        """
        if duration_s <= 0:
            return
        mbps = (num_bytes * 8) / (duration_s * 1e6)
        with self.lock:
            self._transmission_log.append(mbps)
            alpha = self.bandwidth_ema_alpha
            self._bandwidth_mbps = (
                alpha * mbps + (1 - alpha) * self._bandwidth_mbps
            )

    def get_bandwidth_mbps(self) -> float:
        """Current EMA-smoothed uplink bandwidth estimate, B_net."""
        with self.lock:
            return self._bandwidth_mbps

    def get_cpu_utilization(self) -> float:
        """Current CPU utilization, R_cpu, in [0, 1]."""
        with self.lock:
            return self.state['cpu_util']

    def get_state(self) -> Dict:
        """Full diagnostic state (CPU, memory, battery, thermal, bandwidth)."""
        with self.lock:
            state = dict(self.state)
        state['bandwidth_mbps'] = self.get_bandwidth_mbps()
        return state
