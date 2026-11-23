"""Lateral controller for 2020+ Honda CR-V 5G running validated PTM firmware (39990-TLA,A220).

Identical control law, rack map, command delay, model-delay schedule, 20 Hz -> 100 Hz action ramp,
and post-override fade/I-bleed to Biscuit-Labs Honda-PTM-Testing, with zero C++ Params or AOL extras.
"""

import math
from collections import deque

import numpy as np

from openpilot.cereal import log
from opendbc.car.honda.modified_eps import DEFAULT_SETTINGS, ModifiedEpsSettings, get_eps_modified_steering_pressed
from opendbc.car.honda.steer_ratio import get_honda_vgr_inverse, normalize_honda_eps_fw
from opendbc.car.honda.values import CarControllerParams, HondaFlags
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.honda_eps_calibrations import (
  REQUIRED_PTM_RWD,
  select_eps_firmware_calibration,
)
from openpilot.selfdrive.controls.lib.honda_eps_firmware_ff import (
  EpsFirmwareCalibration,
  HondaEpsFirmwareFeedforward,
  HondaEpsLateralCore,
  OVERRIDE_FADE_UP_S_DEFAULT,
  command_delay,
)
from openpilot.selfdrive.controls.lib.honda_eps_rack_map import HondaEpsRackMap
from openpilot.selfdrive.controls.lib.latcontrol import LatControl

HONDA_ANGLE_RATE_LIMIT_DEG_S = 300.0
HONDA_TORQUE_OUTPUT_LPF_TAU = (0.07, 0.05, 0.01)
LAT_SMOOTH_SECONDS = 0.1

CMD_DELAY_SCALE_BY_SOURCE = {"plan": 0.0}
CMD_DELAY_RAMP_S = 1.0


def rate_limit_desired_angle(angle_deg: float, prev_angle_deg: float, max_rate_deg_s: float, dt: float) -> float:
  if max_rate_deg_s <= 0.0 or not math.isfinite(angle_deg):
    return angle_deg
  max_delta = max_rate_deg_s * dt
  return float(min(max(angle_deg, prev_angle_deg - max_delta), prev_angle_deg + max_delta))


class StaticModifiedEpsSettings:
  """Provides DEFAULT_SETTINGS (override_threshold=1700, fade_up_s=0.5, min_steer_speed_mph=1)
  to opendbc's CarController and CarState when LatControlHondaEps is active."""
  def __init__(self, settings: ModifiedEpsSettings = DEFAULT_SETTINGS):
    self._settings = settings

  def current(self) -> ModifiedEpsSettings:
    return self._settings


class CommandDelay:
  """Linearly interpolates the desired curvature issued `delay` seconds ago across frames."""
  def __init__(self, dt: float, max_delay: float):
    self.dt = dt
    self.buf: deque[float] = deque(maxlen=int(math.ceil(max_delay / dt)) + 2)

  def update(self, value: float, delay: float) -> float:
    self.buf.append(float(value))
    if delay <= 0.0:
      return float(value)
    steps = delay / self.dt
    i = int(steps)
    frac = steps - i
    n = len(self.buf)
    newer = self.buf[max(n - 1 - i, 0)]
    older = self.buf[max(n - 2 - i, 0)]
    return newer + frac * (older - newer)


def _is_primary_eps_fw(fw) -> bool:
  is_eps = fw.ecu == car.CarParams.Ecu.eps or str(fw.ecu) == "eps"
  return is_eps and not fw.logging and fw.address == 0x18DA30F1 and fw.subAddress in (None, 0)


def eps_firmware_calibration(CP) -> EpsFirmwareCalibration | None:
  """Returns the validated PTM EPS calibration for CP, or None to keep stock lateral control."""
  if not (CP.brand == "honda" and bool(CP.flags & HondaFlags.EPS_MODIFIED) and CP.lateralTuning.which() == "pid"):
    return None
  eps_fw = next((normalize_honda_eps_fw(fw.fwVersion) for fw in CP.carFw if _is_primary_eps_fw(fw)), None)
  if eps_fw is None:
    return None
  calibration = select_eps_firmware_calibration(str(CP.carFingerprint), eps_fw)
  if calibration is None:
    return None
  cap = float(calibration.e4_per_output)
  ccp = CarControllerParams(CP)
  if [float(x) for x in ccp.STEER_LOOKUP] != [-cap, 0.0, cap] or [float(x) for x in ccp.STEER_LOOKUP_V] != [-cap, 0.0, cap]:
    return None
  if calibration.rack is None or get_honda_vgr_inverse(CP.flags) is None:
    return None
  return calibration


def use_honda_eps_controller(CP) -> bool:
  return eps_firmware_calibration(CP) is not None


def lateral_smooth_seconds(CP, default: float) -> float:
  return LAT_SMOOTH_SECONDS if use_honda_eps_controller(CP) else default


class ModelActionRamp:
  """Ramps modeld's 20 Hz action across each 100 Hz control frame so R6 sees no 20 Hz staircase."""
  def __init__(self):
    self.reset(0.0)

  def reset(self, curvature: float) -> None:
    self.output = self.start = self.target = float(curvature)
    self.elapsed = 0.0

  def update(self, curvature: float, new_frame: bool, active: bool) -> float:
    if not active:
      self.reset(curvature)
      return self.output
    if new_frame:
      self.start, self.target, self.elapsed = self.output, float(curvature), 0.0
    else:
      self.elapsed += DT_CTRL
    self.output = self.start + min(self.elapsed / DT_MDL, 1.0) * (self.target - self.start)
    return self.output


def lateral_delay_schedule(CP) -> tuple | None:
  calibration = eps_firmware_calibration(CP)
  return calibration.lat_delay_schedule if calibration is not None else None


def scheduled_lateral_delay(schedule: tuple, v_ego: float) -> float:
  return float(np.interp(v_ego, *schedule))


def get_rack_map(CP, calibration: EpsFirmwareCalibration | None) -> HondaEpsRackMap | None:
  vgr_inverse = get_honda_vgr_inverse(CP.flags)
  if calibration is None or calibration.rack is None or vgr_inverse is None:
    return None
  return HondaEpsRackMap(CP.wheelbase, vgr_inverse, calibration.rack)


class LatControlHondaEps(LatControl):
  def __init__(self, CP, CI, dt, calibration: EpsFirmwareCalibration | None = None):
    super().__init__(CP, CI, dt)
    calibration = calibration if calibration is not None else eps_firmware_calibration(CP)
    assert calibration is not None, f"no EPS firmware calibration for {CP.carFingerprint}"
    self.calibration: EpsFirmwareCalibration = calibration
    required = REQUIRED_PTM_RWD.get(self.calibration.name)
    cloudlog.info(
      f"LatControlHondaEps: {CP.carFingerprint} steering with EPS calibration {self.calibration.name}"
      + (f"; requires PTM RWD sha256 {required}..." if required else "")
    )
    pid = CP.lateralTuning.pid
    self.core = HondaEpsLateralCore(
      [float(x) for x in pid.kpBP], [float(x) for x in pid.kpV],
      [float(x) for x in pid.kiBP], [float(x) for x in pid.kiV], dt,
      ff=HondaEpsFirmwareFeedforward(dt, cal=self.calibration),
    )
    self.rack_map = get_rack_map(CP, self.calibration)
    assert self.rack_map is not None, f"no rack map for {self.calibration.name}"
    self.cmd_delay = CommandDelay(dt, self.calibration.cmd_delay_s)
    self.lateral_source = "actionHead"
    self.cmd_delay_scale = 1.0
    self.angle_rate_limit_deg_s = HONDA_ANGLE_RATE_LIMIT_DEG_S
    self.core.output_lpf_enabled = True
    self.core.output_lpf_tau = HONDA_TORQUE_OUTPUT_LPF_TAU
    self.core.override_fade_up_s = OVERRIDE_FADE_UP_S_DEFAULT
    self.prev_rate_limited_angle = 0.0
    self.steering_pressed_filter_s = 0.0
    self.steering_pressed_prev = False

  def update_lateral_source(self, source: str) -> None:
    self.lateral_source = str(source)

  def _command_delay(self, v_ego: float) -> float:
    target = CMD_DELAY_SCALE_BY_SOURCE.get(self.lateral_source, 1.0)
    step = self.dt / CMD_DELAY_RAMP_S
    self.cmd_delay_scale += max(min(target - self.cmd_delay_scale, step), -step)
    return self.cmd_delay_scale * command_delay(self.calibration, v_ego)

  def reset(self):
    super().reset()
    self.core.reset()
    self.steering_pressed_filter_s = 0.0
    self.steering_pressed_prev = False

  def _desired_angle_no_offset(self, VM, v_ego, roll, desired_curvature):
    return self.rack_map.angle_from_curvature(desired_curvature, v_ego, roll)

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    pid_log = log.ControlsState.LateralPIDState.new_message()
    pid_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    pid_log.steeringRateDeg = float(CS.steeringRateDeg)

    cmd_delay = self._command_delay(CS.vEgo)
    desired_curvature = self.cmd_delay.update(desired_curvature, cmd_delay)
    angle_des_no_offset = self._desired_angle_no_offset(VM, CS.vEgo, params.roll, desired_curvature)
    if active:
      angle_des_no_offset = rate_limit_desired_angle(angle_des_no_offset, self.prev_rate_limited_angle,
                                                     self.angle_rate_limit_deg_s, self.dt)
    self.prev_rate_limited_angle = angle_des_no_offset

    angle_des = angle_des_no_offset + params.angleOffsetDeg
    pid_log.steeringAngleDesiredDeg = angle_des
    pid_log.angleError = angle_des - CS.steeringAngleDeg

    if not active:
      self.reset()
      output = 0.0
      pid_log.active = False
    else:
      self.steering_pressed_filter_s, steering_pressed = get_eps_modified_steering_pressed(
        bool(CS.steeringPressed), float(getattr(CS, "steeringTorque", 0.0)), float(self.core.output),
        self.steering_pressed_filter_s, self.steering_pressed_prev,
      )
      self.steering_pressed_prev = steering_pressed
      output = self.core.update(angle_des_no_offset, params.angleOffsetDeg, CS.steeringAngleDeg, CS.vEgo, params.roll,
                                steering_pressed, steer_limited_by_safety)
      output = float(max(min(output, self.steer_max), -self.steer_max))

      pid_log.active = True
      pid_log.p = float(self.core.pid.p)
      pid_log.i = float(self.core.pid.i)
      pid_log.f = float(self.core.pid.f)
      pid_log.output = output
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output) < 1e-3, CS, steer_limited_by_safety,
                                                      curvature_limited))

    return output, angle_des, pid_log
