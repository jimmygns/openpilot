import math
import sys
import unittest
from types import ModuleType, SimpleNamespace

import numpy as np

try:
  import openpilot.common.params
except OSError:
  _shim = ModuleType("openpilot.common.params")

  class _FallbackParams:
    def get(self, k, block=False, return_default=False):
      return None

    def get_bool(self, k):
      return False

    def put(self, k, v, block=False):
      pass

    def put_bool(self, k, v, block=False):
      pass

    def put_nonblocking(self, k, v):
      pass

    def remove(self, k):
      pass

  _shim.Params = _FallbackParams  # type: ignore[attr-defined]
  sys.modules["openpilot.common.params"] = _shim

try:
  import openpilot.cereal.messaging  # noqa: F401
except Exception:
  _msg_shim = ModuleType("openpilot.cereal.messaging")
  _msg_shim.SubMaster = object  # type: ignore[attr-defined]
  _msg_shim.PubMaster = object  # type: ignore[attr-defined]
  sys.modules["openpilot.cereal.messaging"] = _msg_shim

from openpilot.cereal import log
from opendbc.can import CANPacker, CANParser
from opendbc.car import DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.modified_eps import DEFAULT_SETTINGS, lane_change_nudge_torque
from opendbc.car.honda.values import CAR
from opendbc.car.honda.yaw_rate import DBC_SCALE, DBC_ZERO
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.honda_eps_calibrations import CRV_TLA_A220, REQUIRED_PTM_RWD
from openpilot.selfdrive.controls.lib.honda_eps_firmware_ff import (
  CLARITY_TRW_A020_LAT_DELAY_SCHEDULE,
  column_load,
  firmware_output,
  friction_width,
  r5_for_motion,
)
from openpilot.selfdrive.controls.lib.honda_eps_rack_map import CRV_TLA_A040_RACK
from openpilot.selfdrive.controls.lib.latcontrol_honda_eps import (
  LAT_SMOOTH_SECONDS,
  LatControlHondaEps,
  ModelActionRamp,
  StaticModifiedEpsSettings,
  eps_firmware_calibration,
  get_rack_map,
  lateral_delay_schedule,
  lateral_smooth_seconds,
  scheduled_lateral_delay,
  use_honda_eps_controller,
)
from openpilot.selfdrive.controls.lib.longcontrol import LongControl

Ecu = structs.CarParams.Ecu
LongCtrlState = structs.CarControl.Actuators.LongControlState


def _eps_fw(version: bytes, address: int = 0x18DA30F1, sub: int = 0, logging: bool = False):
  return structs.CarParams.CarFw(
    ecu=Ecu.eps, address=address, subAddress=sub, fwVersion=version, brand="honda", logging=logging
  )


def _crv_params(firmware, alpha_long: bool = True):
  fp = gen_empty_fingerprint()
  initial = CarInterface.get_params(CAR.HONDA_CRV_5G, fp, [], alpha_long, False, False)
  fp[CanBus(initial).pt][0x191] = 8
  fp[CanBus(initial).pt][0x1BE] = 3
  cp = CarInterface.get_params(CAR.HONDA_CRV_5G, fp, firmware, alpha_long, False, False)
  cp.carFw = firmware
  return cp


class TestCrv2020PtmLateralAndSafety(unittest.TestCase):
  def test_calibration_gating_and_fail_closed_safety(self):
    ptm_cp = _crv_params([_eps_fw(b"39990-TLA,A220\x00\x00"), _eps_fw(b"39990-LOGGING,0000\x00\x00", logging=True)])
    stock_cp = _crv_params([_eps_fw(b"39990-TLA-A220\x00\x00")])
    unknown_ptm_cp = _crv_params([_eps_fw(b"39990-TLA,A040\x00\x00")])

    self.assertEqual(eps_firmware_calibration(ptm_cp), CRV_TLA_A220)
    self.assertTrue(use_honda_eps_controller(ptm_cp))
    self.assertEqual(REQUIRED_PTM_RWD["crv_tla_a220"], "2c992997")
    self.assertAlmostEqual(CRV_TLA_A220.output_ceiling(), 3841.0 / 4096.0, places=9)

    # Stock CR-V and unknown PTM builds must never activate LatControlHondaEps
    self.assertIsNone(eps_firmware_calibration(stock_cp))
    self.assertFalse(use_honda_eps_controller(stock_cp))
    self.assertTrue(unknown_ptm_cp.dashcamOnly)
    self.assertIsNone(eps_firmware_calibration(unknown_ptm_cp))

  def test_road_measured_rack_map_and_delay_schedules(self):
    ptm_cp = _crv_params([_eps_fw(b"39990-TLA,A220\x00\x00")])
    stock_cp = _crv_params([_eps_fw(b"39990-TLA-A220\x00\x00")])

    self.assertEqual(CRV_TLA_A220.rack, CRV_TLA_A040_RACK)
    self.assertEqual(CRV_TLA_A040_RACK.slip_factor, -0.00084)
    self.assertEqual(lateral_smooth_seconds(ptm_cp, 0.0), LAT_SMOOTH_SECONDS)
    self.assertEqual(lateral_smooth_seconds(stock_cp, 0.0), 0.0)
    self.assertEqual(lateral_delay_schedule(ptm_cp), CLARITY_TRW_A020_LAT_DELAY_SCHEDULE)
    self.assertIsNone(lateral_delay_schedule(stock_cp))
    self.assertAlmostEqual(scheduled_lateral_delay(CLARITY_TRW_A020_LAT_DELAY_SCHEDULE, 7.0), 0.08)
    self.assertAlmostEqual(scheduled_lateral_delay(CLARITY_TRW_A020_LAT_DELAY_SCHEDULE, 20.0), 0.20)

    rack = get_rack_map(ptm_cp, CRV_TLA_A220)
    assert rack is not None
    for v_ego in (3.0, 12.0, 28.0):
      for roll in (-0.05, 0.0, 0.05):
        for angle in (-320.0, -90.0, -15.0, 0.0, 15.0, 90.0, 320.0):
          curv = rack.curvature_from_angle(angle, v_ego, roll)
          recovered = rack.angle_from_curvature(curv, v_ego, roll)
          self.assertAlmostEqual(recovered, angle, places=5)

    # ModelActionRamp linearly interpolates 20 Hz model updates across 5 x 100 Hz control steps
    ramp = ModelActionRamp()
    self.assertEqual(ramp.update(0.01, new_frame=True, active=False), 0.01)
    vals = [ramp.update(0.02, new_frame=(step == 0), active=True) for step in range(6)]
    expected = [0.01 + min(step * DT_CTRL / DT_MDL, 1.0) * 0.01 for step in range(6)]
    np.testing.assert_allclose(vals, expected, atol=1e-12)

  def test_firmware_law_exact_quadratic_inversion(self):
    for v_ego in (4.0, 15.0, 30.0):
      w = friction_width(v_ego, CRV_TLA_A220)
      for angle in (-180.0, -35.0, 0.0, 35.0, 180.0):
        for rate in (-180.0, -40.0, 0.0, 40.0, 180.0):
          target_load = column_load(angle, rate, v_ego, 0.02, CRV_TLA_A220.load, w)
          r5 = r5_for_motion(target_load, rate, CRV_TLA_A220, 0.0, angle)
          delivered = firmware_output(r5, rate, CRV_TLA_A220, angle)
          self.assertAlmostEqual(delivered, target_load, places=6)

  def test_desire_helper_lane_change_nudge_at_stock_threshold(self):
    ptm_cp = _crv_params([_eps_fw(b"39990-TLA,A220\x00\x00")])
    nudge = lane_change_nudge_torque(ptm_cp)
    self.assertEqual(nudge, 1200)

    dh = DesireHelper(nudge_torque=nudge)
    cs_signal = SimpleNamespace(
      vEgo=25.0, leftBlinker=True, rightBlinker=False,
      steeringPressed=False, steeringTorque=0.0,
      leftBlindspot=False, rightBlindspot=False,
    )
    dh.update(cs_signal, lateral_active=True, lane_change_prob=0.5)
    self.assertEqual(dh.lane_change_state, log.LaneChangeState.preLaneChange)

    # Gentle 1350 driver nudge (< 1700 override threshold, so steeringPressed stays False) confirms lane change
    cs_nudge = SimpleNamespace(
      vEgo=25.0, leftBlinker=True, rightBlinker=False,
      steeringPressed=False, steeringTorque=1350.0,
      leftBlindspot=False, rightBlindspot=False,
    )
    dh.update(cs_nudge, lateral_active=True, lane_change_prob=0.5)
    self.assertEqual(dh.lane_change_state, log.LaneChangeState.laneChangeStarting)
    self.assertEqual(dh.desire, log.Desire.laneChangeLeft)

  def test_post_override_fade_and_integrator_bleed(self):
    ptm_cp = _crv_params([_eps_fw(b"39990-TLA,A220\x00\x00")])
    ci = CarInterface(ptm_cp)
    settings = StaticModifiedEpsSettings()
    ci.CC.modified_eps_settings = settings
    ci.CS.modified_eps_settings = settings
    self.assertEqual(settings.current(), DEFAULT_SETTINGS)

    lac = LatControlHondaEps(ptm_cp, ci, DT_CTRL)
    lp = SimpleNamespace(roll=0.0, angleOffsetDeg=0.0)

    # Build steady turn integrator state
    for _ in range(150):
      cs = SimpleNamespace(vEgo=15.0, steeringAngleDeg=10.0, steeringRateDeg=0.0, steeringPressed=False, steeringTorque=200.0)
      steer, _, _ = lac.update(True, cs, None, lp, False, -0.01, False, 0.2)
    i_before_press = lac.core.pid.i
    self.assertGreater(abs(i_before_press), 0.002)
    self.assertGreater(abs(steer), 0.02)

    # Driver override: cuts FF ramp to 0 and holds integrator
    cs_press = SimpleNamespace(vEgo=15.0, steeringAngleDeg=10.0, steeringRateDeg=0.0, steeringPressed=True, steeringTorque=2300.0)
    lac.update(True, cs_press, None, lp, True, -0.01, False, 0.2)
    self.assertEqual(lac.core.ff_ramp, 0.0)
    self.assertAlmostEqual(lac.core.pid.i, i_before_press, places=9)

    # Post-override 0.5 s fade (steer_limited=True, steeringPressed=False): bleeds held I toward 0
    cs_release = SimpleNamespace(vEgo=15.0, steeringAngleDeg=10.0, steeringRateDeg=0.0, steeringPressed=False, steeringTorque=300.0)
    for _ in range(50):
      lac.update(True, cs_release, None, lp, True, -0.01, False, 0.2)
    self.assertAlmostEqual(lac.core.pid.i, i_before_press * math.exp(-0.5 / 0.5), places=6)

  def test_longitudinal_100_percent_bit_exact_with_stock_mvl(self):
    class _MemParams:
      def __init__(self):
        self.store = {}

      def get(self, k):
        return self.store.get(k)

      def put(self, k, v, block=False):
        self.store[k] = v

      def put_nonblocking(self, k, v):
        self.store[k] = v

    import opendbc.car.honda.carcontroller as honda_cc
    orig_params = honda_cc.Params
    honda_cc.Params = _MemParams
    try:
      ptm_cp = _crv_params([_eps_fw(b"39990-TLA,A220\x00\x00")], alpha_long=True)
      stock_cp = _crv_params([_eps_fw(b"39990-TLA-A220\x00\x00")], alpha_long=True)

      ci_ptm = CarInterface(ptm_cp)
      ci_stock = CarInterface(stock_cp)
      eps_settings = StaticModifiedEpsSettings()
      ci_ptm.CC.modified_eps_settings = eps_settings
      ci_ptm.CS.modified_eps_settings = eps_settings

      loc_ptm = LongControl(ptm_cp)
      loc_stock = LongControl(stock_cp)

      ci_ptm.update([])
      ci_stock.update([])

      pt_bus = CanBus(ptm_cp).pt
      packer = CANPacker("honda_civic_hatchback_ex_2017_can_generated")
      pt_parser_ptm = CANParser(
        "honda_civic_hatchback_ex_2017_can_generated",
        [("ACC_CONTROL", 50), ("ACC_HUD", 10)],
        pt_bus,
      )
      pt_parser_stock = CANParser(
        "honda_civic_hatchback_ex_2017_can_generated",
        [("ACC_CONTROL", 50), ("ACC_HUD", 10)],
        pt_bus,
      )

      # Simulate 500 frames across standstill -> launch -> cruise -> decel -> stop
      for step in range(500):
        kph = 0.0 if step < 50 else min(80.0, (step - 50) * 0.4) if step < 300 else max(0.0, 80.0 - (step - 300) * 0.5)
        a_target = -0.8 if step < 50 else 1.2 if step < 200 else 0.0 if step < 300 else -1.5
        should_stop = step < 50 or step > 450

        msgs = [
          packer.make_can_msg("ENGINE_DATA", pt_bus, {"XMISSION_SPEED": kph, "ENGINE_RPM": 1500}),
          packer.make_can_msg("WHEEL_SPEEDS", pt_bus, {
            "WHEEL_SPEED_FL": kph, "WHEEL_SPEED_FR": kph, "WHEEL_SPEED_RL": kph, "WHEEL_SPEED_RR": kph,
          }),
          packer.make_can_msg("SCM_FEEDBACK", pt_bus, {"MAIN_ON": 1}),
          packer.make_can_msg("POWERTRAIN_DATA", pt_bus, {"ACC_STATUS": 1}),
          packer.make_can_msg("STEER_STATUS", pt_bus, {"STEER_TORQUE_SENSOR": 200, "STEER_CONTROL_ACTIVE": 1}),
          packer.make_can_msg("KINEMATICS", pt_bus, {"YAW_RATE": (510 - DBC_ZERO) * DBC_SCALE}),
        ]
        t_ns = step * 10_000_000
        cs_ptm = ci_ptm.update([(t_ns, msgs)])
        cs_stock = ci_stock.update([(t_ns, msgs)])
        cs_ptm.vCruise = 90.0
        cs_stock.vCruise = 90.0

        lim_ptm = ci_ptm.get_pid_accel_limits(ptm_cp, cs_ptm.vEgo, 25.0)
        lim_stock = ci_stock.get_pid_accel_limits(stock_cp, cs_stock.vEgo, 25.0)
        self.assertEqual(lim_ptm, lim_stock)

        accel_ptm = loc_ptm.update(True, cs_ptm, a_target, should_stop, lim_ptm)
        accel_stock = loc_stock.update(True, cs_stock, a_target, should_stop, lim_stock)
        self.assertEqual(accel_ptm, accel_stock)
        self.assertEqual(loc_ptm.long_control_state, loc_stock.long_control_state)

        cc_ptm = structs.CarControl()
        cc_ptm.enabled = True
        cc_ptm.latActive = True
        cc_ptm.longActive = True
        cc_ptm.actuators.accel = float(accel_ptm)
        cc_ptm.actuators.longControlState = loc_ptm.long_control_state
        cc_ptm.actuators.torque = 0.35 * math.sin(step * 0.05)
        cc_ptm.hudControl.setSpeed = 25.0
        cc_ptm.hudControl.speedVisible = True

        cc_stock = structs.CarControl()
        cc_stock.enabled = True
        cc_stock.latActive = True
        cc_stock.longActive = True
        cc_stock.actuators.accel = float(accel_stock)
        cc_stock.actuators.longControlState = loc_stock.long_control_state
        cc_stock.actuators.torque = 0.35 * math.sin(step * 0.05)
        cc_stock.hudControl.setSpeed = 25.0
        cc_stock.hudControl.speedVisible = True

        out_ptm, sends_ptm = ci_ptm.apply(cc_ptm.as_reader(), t_ns)
        out_stock, sends_stock = ci_stock.apply(cc_stock.as_reader(), t_ns)

        # Every longitudinal actuator and adaptive learner state is 100% bit-for-bit identical
        self.assertEqual(out_ptm.accel, out_stock.accel)
        self.assertEqual(out_ptm.gas, out_stock.gas)
        self.assertEqual(out_ptm.brake, out_stock.brake)
        self.assertEqual(out_ptm.speed, out_stock.speed)

        # Every non-steering CAN frame (ACC_CONTROL 0x1DF, ACC_HUD 0x30C, RADAR_HUD 0x300, tester present) is byte-identical
        long_frames_ptm = [(addr, dat, bus) for addr, dat, bus in sends_ptm if addr != 0xE4]
        long_frames_stock = [(addr, dat, bus) for addr, dat, bus in sends_stock if addr != 0xE4]
        self.assertEqual(long_frames_ptm, long_frames_stock, f"longitudinal CAN mismatch at frame {step}")

        if any(addr == 0x1DF for addr, _, _ in long_frames_ptm):
          pt_parser_ptm.update([(t_ns, long_frames_ptm)])
          pt_parser_stock.update([(t_ns, long_frames_stock)])
          self.assertEqual(dict(pt_parser_ptm.vl["ACC_CONTROL"]), dict(pt_parser_stock.vl["ACC_CONTROL"]))
    finally:
      honda_cc.Params = orig_params


class TestOfflineCommaBlock(unittest.TestCase):
  def test_api_host_and_api_get_hard_blocked(self):
    from openpilot.common import api

    self.assertEqual(api.API_HOST, "")
    with self.assertRaises(RuntimeError):
      api.api_get("v1.4/0000000000000000/upload_url/")
    client = api.Api("0000000000000000")
    with self.assertRaises(RuntimeError):
      client.get("v1.1/devices/0000000000000000")
    with self.assertRaises(RuntimeError):
      client.post("v2/pilotauth/")

  def test_athena_and_uploader_disabled_in_process_config(self):
    from openpilot.system.manager.process_config import managed_processes

    self.assertFalse(managed_processes["manage_athenad"].enabled)
    self.assertFalse(managed_processes["uploader"].enabled)

  def test_offline_registration_never_contacts_comma(self):
    import hashlib
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from openpilot.system.athena import registration

    store: dict[str, object] = {}

    class _MemParams:
      def get(self, k, block=False, return_default=False):
        return store.get(k)

      def put(self, k, v, block=False):
        store[k] = v

      def remove(self, k):
        store.pop(k, None)

    with tempfile.TemporaryDirectory() as tmp:
      persist = Path(tmp) / "comma"
      persist.mkdir(parents=True)
      (persist / "id_rsa").write_text("fake-priv")
      (persist / "id_rsa.pub").write_text("fake-pub")

      with patch.object(registration, "Params", _MemParams), \
           patch.object(registration, "set_offroad_alert"), \
           patch.object(registration.Paths, "persist_root", return_value=tmp), \
           patch("openpilot.common.api.Paths.persist_root", return_value=tmp):
        # 1. Device without cached dongle_id derives deterministic local 16-hex ID offline
        with patch.object(registration, "PC", False), \
             patch.object(registration.HARDWARE, "get_serial", return_value="c3x-ptm-999"):
          expected = hashlib.sha256(b"local-c3x-ptm-999").hexdigest()[:16]
          self.assertEqual(registration.register(), expected)
          self.assertEqual(store["DongleId"], expected)

        # 2. Cached /persist/comma/dongle_id is preserved offline
        store.clear()
        (persist / "dongle_id").write_text("a1b2c3d4e5f60718\n")
        self.assertEqual(registration.register(), "a1b2c3d4e5f60718")
        self.assertEqual(store["DongleId"], "a1b2c3d4e5f60718")

  def test_uploader_hard_disabled_even_if_called_directly(self):
    import tempfile
    from pathlib import Path

    from openpilot.system.loggerd import uploader as uploader_mod

    with tempfile.TemporaryDirectory() as tmp:
      seg = Path(tmp) / "00000013--a2b85a2f33--0"
      seg.mkdir()
      (seg / "qlog").write_bytes(b"ptm-drive-data")

      up = uploader_mod.Uploader("0000000000000000", tmp)
      with self.assertRaises(RuntimeError):
        up.do_upload("00000013--a2b85a2f33--0/qlog.zst", str(seg / "qlog"))
      self.assertFalse(up.step(1, False))
      # Direct main() invocation exits immediately as a no-op
      self.assertIsNone(uploader_mod.main())


if __name__ == "__main__":
  unittest.main()
