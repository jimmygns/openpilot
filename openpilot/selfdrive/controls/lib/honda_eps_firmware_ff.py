"""Modified-EPS Honda lateral control law: EPS firmware R5/R6 rate-damping inversion + residual angle PID."""

import math
from dataclasses import dataclass, field

import numpy as np

from opendbc.car.honda.steer_ratio import HONDA_VGR_CENTRE_DIVISOR_BY_PROFILE, HONDA_VGR_INVERSE_BY_PROFILE
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.honda_eps_rack_map import RackMapTable


def command_key(e4: float) -> int:
  """0xE4 value -> command-map key, with the no-SHLL2 decode (every PTM image)."""
  return int(math.trunc(math.trunc(e4 * 56756 / 32768) / 4))


@dataclass(frozen=True)
class ColumnLoadModel:
  """Firmware output counts (A030 sign convention) a motion needs:
    load = k0*th + k1*th*v^2 + c*thd + friction*tanh(thd/w) + bias + kroll*roll*v^2
  th/thd: steering-wheel deg, deg/s; v: m/s; roll: rad."""
  k0: float
  k1: float
  c: float
  friction: float
  bias: float
  kroll: float


CMD_DELAY_DEFAULT_S = 0.12
CMD_DELAY_SPEED_BP = (10.0, 15.0)  # m/s: cmd_delay_s below, cmd_delay_highway_s above


def command_delay(cal: "EpsFirmwareCalibration", v_ego: float) -> float:
  return float(np.interp(v_ego, CMD_DELAY_SPEED_BP, (cal.cmd_delay_s, cal.cmd_delay_highway_s)))


CLARITY_TRW_A020_LAT_DELAY_SCHEDULE = ((3.5, 7.0, 12.0, 20.0, 30.0), (0.15, 0.08, 0.10, 0.20, 0.30))  # m/s, s
LAT_DELAY_SCHEDULE_DEFAULT = CLARITY_TRW_A020_LAT_DELAY_SCHEDULE


@dataclass(frozen=True)
class EpsFirmwareCalibration:
  """Per-image constants of the firmware-inversion chain, plus the car's residual-PID trims."""
  name: str
  e4_per_output: float
  r5_key_bp: tuple
  r5_v: tuple
  key_clamp: float
  envelope_bp: tuple
  envelope_v: tuple
  kp_key_bp: tuple
  kp_v: tuple
  r6_per_deg_s: float
  load: ColumnLoadModel
  p_scale: tuple
  i_scale: tuple
  kff: float = 45.0
  scale_q8: float = 256.0
  r5_per_key: float | None = None
  r6_angle_bp: tuple | None = None
  r6_angle_gain: tuple | None = None
  cmd_delay_s: float = CMD_DELAY_DEFAULT_S
  cmd_delay_highway_s: float = 0.0
  lat_delay_schedule: tuple = LAT_DELAY_SCHEDULE_DEFAULT
  rack: RackMapTable | None = None
  friction_width_v: tuple | None = None
  p_clamp: float | None = None
  kp_pieces: tuple = field(init=False, repr=False, compare=False)

  def __post_init__(self):
    assert self.p_clamp is None or self.kff == 0.0, "a P clamp with KFF is not modeled: no such image"
    if self.r5_per_key is not None:
      pieces = [(lo * self.r5_per_key, hi * self.r5_per_key, kp_lo, (kp_hi - kp_lo) / ((hi - lo) * self.r5_per_key))
                for lo, hi, kp_lo, kp_hi in zip(self.kp_key_bp[:-1], self.kp_key_bp[1:], self.kp_v[:-1], self.kp_v[1:],
                                                strict=True)]
      pieces.append((self.kp_key_bp[-1] * self.r5_per_key, math.inf, self.kp_v[-1], 0.0))
    else:
      knots = sorted(set(self.r5_v) | {float(np.interp(k, self.r5_key_bp, self.r5_v)) for k in self.kp_key_bp
                                       if k <= self.r5_key_bp[-1]})
      kps = [self._kp_at_r5(r) for r in knots]
      pieces = [(lo, hi, kp_lo, (kp_hi - kp_lo) / (hi - lo))
                for lo, hi, kp_lo, kp_hi in zip(knots[:-1], knots[1:], kps[:-1], kps[1:], strict=True)]
      pieces.append((knots[-1], math.inf, kps[-1], 0.0))
    object.__setattr__(self, "kp_pieces", tuple(pieces))

  def _kp_at_r5(self, r5_abs: float) -> float:
    if self.r5_per_key is not None:
      return float(np.interp(r5_abs / self.r5_per_key, self.kp_key_bp, self.kp_v))
    return float(np.interp(np.interp(r5_abs, self.r5_v, self.r5_key_bp), self.kp_key_bp, self.kp_v))

  def key_ceiling(self, v_ego: float) -> float:
    return min(float(np.interp(v_ego * 3.6 * 2.0, self.envelope_bp, self.envelope_v)), self.key_clamp)

  def r5_ceiling(self, v_ego: float) -> float:
    return float(np.interp(self.key_ceiling(v_ego), self.r5_key_bp, self.r5_v))

  def output_ceiling(self) -> float:
    key = min(self.key_clamp, self.r5_key_bp[-1])
    return min(1.0, math.ceil(key * 4.0 * 32768.0 / 56756.0) / self.e4_per_output)


R6_PER_RAW_DEG_S = -122.0  # NORM 1650
R6_CLAMP = 32765.0
R6_GAIN_BP = (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 180, 210, 240, 270, 300, 360, 420, 480)


def computed_r6(profile: str, norm: float = 1650.0) -> tuple[float, tuple, tuple]:
  """(R6 per published deg/s at center, |angle| axis, gain by angle) from an image's A table."""
  linear_bp, angle_bp = (np.asarray(x, dtype=float) for x in HONDA_VGR_INVERSE_BY_PROFILE[profile])

  def linear(a: float) -> float:
    return float(np.interp(a, angle_bp, linear_bp)) if a <= angle_bp[-1] else float(a * linear_bp[-1] / angle_bp[-1])

  def slope(a: float) -> float:
    lo = max(a - 10.0, 0.0)
    return (linear(a + 10.0) - linear(lo)) / (a + 10.0 - lo)

  center = R6_PER_RAW_DEG_S * norm / 1650.0 * HONDA_VGR_CENTRE_DIVISOR_BY_PROFILE[profile] / 16384.0
  return round(center, 2), R6_GAIN_BP, tuple(round(slope(a) / slope(0.0), 4) for a in R6_GAIN_BP)


FRICTION_WIDTH_DEG_S = 5.0
FRICTION_WIDTH_SPEED_BP = [8.0, 15.0]  # m/s
FRICTION_WIDTH_V = [20.0, FRICTION_WIDTH_DEG_S]  # deg/s


def friction_width(v_ego: float, cal: "EpsFirmwareCalibration | None" = None) -> float:
  widths = cal.friction_width_v if cal is not None and cal.friction_width_v is not None else FRICTION_WIDTH_V
  return float(np.interp(v_ego, FRICTION_WIDTH_SPEED_BP, widths))


DESIRED_RATE_TAU = 0.10
LEAD_S = 0.07
FF_OUTPUT_TAU = 0.15
R5_CAP = 27000.0
R5_CAP_ENVELOPE_FRAC = 0.9

MPH_TO_MS = 0.44704
BAND_LOW_MAX = 25.0 * MPH_TO_MS
BAND_STD_MAX = 50.0 * MPH_TO_MS
OUTPUT_LPF_TAU = (0.07, 0.05, 0.01)
INTEGRATOR_MIN_SPEED = 2.0
OVERRIDE_FADE_I_BLEED_TAU = 0.5
OVERRIDE_FADE_UP_S_DEFAULT = 0.5

FF_JOIN_ERROR_DEG = 10.0
FF_FADE_IN_S = 0.5
FF_SPEED_BP = [2.0, 4.0]
FF_CRAWL_ANGLE_BP = [5.0, 20.0]
FF_CRAWL_SPEED_BP = [5.0, 8.0]


def key_ceiling(v_ego: float, cal: EpsFirmwareCalibration) -> float:
  return cal.key_ceiling(v_ego)


def r5_from_output(output: float, v_ego: float, cal: EpsFirmwareCalibration) -> float:
  key = command_key(-output * cal.e4_per_output)
  mag = float(np.interp(min(abs(key), cal.key_ceiling(v_ego)), cal.r5_key_bp, cal.r5_v))
  return math.copysign(mag, key) if key else 0.0


def output_from_r5(r5: float, cal: EpsFirmwareCalibration) -> float:
  key = float(np.interp(min(abs(r5), cal.r5_v[-1]), cal.r5_v, cal.r5_key_bp))
  e4 = key * 4.0 * 32768.0 / 56756.0
  return -math.copysign(e4, r5) / cal.e4_per_output


def firmware_kp(r5: float, cal: EpsFirmwareCalibration) -> float:
  return cal._kp_at_r5(abs(r5))


def firmware_r6(steering_rate_deg_s: float, angle_deg: float, cal: EpsFirmwareCalibration) -> float:
  gain = 1.0 if cal.r6_angle_bp is None or cal.r6_angle_gain is None else float(np.interp(abs(angle_deg), cal.r6_angle_bp, cal.r6_angle_gain))
  return max(min(cal.r6_per_deg_s * gain * steering_rate_deg_s, R6_CLAMP), -R6_CLAMP)


def firmware_law(r5: float, r6: float, kp: float, cal: EpsFirmwareCalibration) -> float:
  p = kp * (r5 - r6)
  if cal.p_clamp is not None:
    p = max(min(p, 1024.0 * cal.p_clamp), -1024.0 * cal.p_clamp)
  return cal.scale_q8 * (p + cal.kff * r5) / 1024.0 / 256.0


def firmware_output(r5: float, steering_rate_deg_s: float, cal: EpsFirmwareCalibration, angle_deg: float = 0.0) -> float:
  return firmware_law(r5, firmware_r6(steering_rate_deg_s, angle_deg, cal), firmware_kp(r5, cal), cal)


def column_load(angle_deg: float, rate_deg_s: float, v_ego: float, roll: float,
                load: ColumnLoadModel, friction_width: float = FRICTION_WIDTH_DEG_S) -> float:
  return (load.k0 * angle_deg + load.k1 * angle_deg * v_ego ** 2 + load.c * rate_deg_s
          + load.friction * math.tanh(rate_deg_s / friction_width) + load.bias + load.kroll * roll * v_ego ** 2)


def r5_for_motion(load: float, rate_deg_s: float, cal: EpsFirmwareCalibration, r5_guess: float = 0.0, angle_deg: float = 0.0) -> float:
  """Solve the piecewise-quadratic firmware law for R5 yielding `load` at `rate_deg_s` through `angle_deg`."""
  x = 1024.0 * load * 256.0 / cal.scale_q8
  if cal.p_clamp is not None:
    x = max(min(x, 1024.0 * cal.p_clamp), -1024.0 * cal.p_clamp)
  r6 = firmware_r6(rate_deg_s, angle_deg, cal)
  roots = []
  for lo, hi, kp_lo, slope in cal.kp_pieces:
    for side in (1.0, -1.0):
      a, b = kp_lo - slope * lo, slope * side
      qa, qb, qc = b, a + cal.kff - b * r6, -(a * r6 + x)
      if abs(qa) < 1e-12:
        candidates = [-qc / qb]
      else:
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:
          continue
        candidates = [(-qb + sq) / (2.0 * qa) for sq in (math.sqrt(disc), -math.sqrt(disc))]
      roots += [r for r in candidates if lo <= side * r <= hi]
  return min(roots, key=lambda r: abs(r - r5_guess))


class HondaEpsFirmwareFeedforward:
  def __init__(self, dt: float, cal: EpsFirmwareCalibration, rate_tau: float = DESIRED_RATE_TAU, lead_s: float = LEAD_S,
               output_tau: float = FF_OUTPUT_TAU, friction_width: float | None = None):
    self.dt = dt
    self.alpha = dt / (rate_tau + dt)
    self.output_alpha = dt / (output_tau + dt)
    self.lead_s = lead_s
    self.friction_width = friction_width
    self.cal = cal
    self.reset()

  def reset(self):
    self.prev_angle = None
    self.rate = 0.0
    self.r5 = 0.0
    self.load = 0.0
    self.output = 0.0

  def update(self, desired_angle_no_offset: float, v_ego: float, roll: float) -> float:
    first = self.prev_angle is None
    if not first:
      raw_rate = (desired_angle_no_offset - self.prev_angle) / self.dt
      self.rate += self.alpha * (max(min(raw_rate, 400.0), -400.0) - self.rate)
    self.prev_angle = desired_angle_no_offset

    angle = desired_angle_no_offset + self.lead_s * self.rate
    width = self.friction_width if self.friction_width is not None else friction_width(v_ego, self.cal)
    self.load = column_load(angle, self.rate, v_ego, roll, self.cal.load, width)
    cap = min(R5_CAP, R5_CAP_ENVELOPE_FRAC * self.cal.r5_ceiling(v_ego))
    self.r5 = max(min(r5_for_motion(self.load, self.rate, self.cal, self.r5, angle), cap), -cap)
    target = output_from_r5(self.r5, self.cal)
    self.output = target if first else self.output + self.output_alpha * (target - self.output)
    return self.output


def speed_band(v_ego: float, values):
  return values[0] if v_ego < BAND_LOW_MAX else values[1] if v_ego < BAND_STD_MAX else values[2]


class EpsPIDController(PIDController):
  """PIDController with speed-band integrator gain scaling and low-speed integrator reset,
  kept local to HondaEpsLateralCore so MVL's shared LongControl PIDController stays untouched."""
  def update(self, error, error_rate=0.0, speed=0.0, feedforward=0., freeze_integrator=False,
             integrator_gain_scale=1.0, reset_integrator=False):
    self.speed = speed
    self.p = float(self.k_p) * float(error)
    self.d = float(self.k_d) * error_rate
    self.f = feedforward

    if reset_integrator:
      self.i = 0.0

    if not freeze_integrator:
      i = self.i + float(self.k_i) * float(integrator_gain_scale) * self.i_dt * float(error)
      test_control = self.p + i + self.d + self.f
      i_upperbound = self.i if test_control > self.pos_limit else self.pos_limit
      i_lowerbound = self.i if test_control < self.neg_limit else self.neg_limit
      self.i = float(min(max(i, i_lowerbound), i_upperbound))

    control = self.p + self.i + self.d + self.f
    self.control = float(min(max(control, self.neg_limit), self.pos_limit))
    return self.control


class HondaEpsLateralCore:
  """Angle PID on the residual + firmware-inversion feedforward, faded in once the wheel is on the path."""

  output_lpf_enabled = True
  output_lpf_tau = OUTPUT_LPF_TAU

  def __init__(self, kp_bp, kp_v, ki_bp, ki_v, dt: float, ff: HondaEpsFirmwareFeedforward, p_scale=None, i_scale=None):
    self.dt = dt
    self.pid = EpsPIDController((kp_bp, kp_v), (ki_bp, ki_v), pos_limit=1.0, neg_limit=-1.0, rate=1.0 / dt)
    self.ff = ff
    self.p_scale = p_scale if p_scale is not None else self.ff.cal.p_scale
    self.i_scale = i_scale if i_scale is not None else self.ff.cal.i_scale
    self.output_lpf = FirstOrderFilter(0.0, OUTPUT_LPF_TAU[0], dt)
    self.override_fade_up_s = OVERRIDE_FADE_UP_S_DEFAULT
    self.reset()

  def reset(self):
    self.pid.reset()
    self.ff.reset()
    self.since_press_s = math.inf
    self.ff_ramp = 0.0
    self.ff_weight = 0.0
    self.output_lpf.x = 0.0
    self.output_lpf.initialized = True
    self.output = 0.0

  def update(self, desired_angle_no_offset: float, angle_offset: float, angle: float, v_ego: float, roll: float,
             steering_pressed: bool, steer_limited: bool) -> float:
    error = desired_angle_no_offset + angle_offset - angle
    ff_full = self.ff.update(desired_angle_no_offset, v_ego, roll)

    if steering_pressed or v_ego < FF_SPEED_BP[0]:
      self.ff_ramp = 0.0
    elif self.ff_ramp > 0.0 or abs(error) < FF_JOIN_ERROR_DEG:
      self.ff_ramp = min(1.0, self.ff_ramp + self.dt / FF_FADE_IN_S)
    crawl = float(np.interp(v_ego, FF_CRAWL_SPEED_BP, [1.0, 0.0]))
    crawl_gate = 1.0 - crawl * (1.0 - float(np.interp(abs(desired_angle_no_offset), FF_CRAWL_ANGLE_BP, [0.0, 1.0])))
    self.ff_weight = self.ff_ramp * float(np.interp(v_ego, FF_SPEED_BP, [0.0, 1.0])) * crawl_gate
    ff = self.ff_weight * ff_full

    self.since_press_s = 0.0 if steering_pressed else self.since_press_s + self.dt
    if steer_limited and not steering_pressed and self.since_press_s <= self.override_fade_up_s + self.dt:
      self.pid.i *= math.exp(-self.dt / OVERRIDE_FADE_I_BLEED_TAU)

    limit = self.ff.cal.output_ceiling()
    self.pid.set_limits(limit, -limit)
    i_scale = speed_band(v_ego, self.i_scale)
    self.pid.update(error, speed=v_ego, feedforward=ff,
                    freeze_integrator=steer_limited or steering_pressed or v_ego < INTEGRATOR_MIN_SPEED,
                    integrator_gain_scale=i_scale,
                    reset_integrator=i_scale <= 0.0 or v_ego < INTEGRATOR_MIN_SPEED)
    output = max(min(self.pid.p * speed_band(v_ego, self.p_scale) + self.pid.i + self.pid.d + ff, limit), -limit)

    if self.output_lpf_enabled:
      self.output_lpf.update_alpha(speed_band(v_ego, self.output_lpf_tau))
      output = float(self.output_lpf.update(output))
    else:
      self.output_lpf.x = output
      self.output_lpf.initialized = True
    self.output = max(min(output, limit), -limit)
    return self.output
