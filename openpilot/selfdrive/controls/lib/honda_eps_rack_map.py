"""Wheel angle <-> curvature through CR-V 5G's VSA-yaw-identified rack ratio (Honda-PTM-Testing)."""

import math
from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81


@dataclass(frozen=True)
class RackMapTable:
  """Effective rack ratio after the firmware angle table, by physical wheel angle, and the slip factor."""
  ratio_bp: tuple[float, ...]  # physical wheel angle, deg
  ratio_v: tuple[float, ...]
  slip_factor: float           # 1 / (m/s)^2


# CR-V 5G (TLA-A040 / TLA-A220 PTM rev b), fitted against VSA yaw (0x94 bus 1) over 994 s moving at >= 3 deg,
# and validated on CR-V A220's own VSA yaw across routes 0000030a, 0000030d, 00000039 and 0000003d (740 s):
# measured / predicted curvature 0.95 / 0.99 at 25-40 m/s and 0.97 / 1.01 at 15-25 m/s (vs 0.79 / 0.90 on spec map).
CRV_TLA_A040_RACK = RackMapTable(
  ratio_bp=(5.4, 13.5, 25.8, 55.4, 85.2, 124.9, 173.4, 228.1, 302.5, 376.8, 414.0),
  ratio_v=(17.38, 16.91, 16.67, 16.46, 16.25, 16.17, 16.04, 15.86, 15.52, 15.09, 14.83),
  slip_factor=-0.00084,
)


class HondaEpsRackMap:
  """Physical wheel angle (deg, left-positive) <-> curvature (1/m, openpilot's right-positive) for one RackMapTable."""
  def __init__(self, wheelbase: float, vgr_inverse, table: RackMapTable):
    linear_bp, angle_bp = (np.asarray(x, dtype=float) for x in vgr_inverse)
    self.wheelbase = float(wheelbase)
    self.slip_factor = table.slip_factor
    self.angle_grid = np.unique(np.r_[angle_bp, np.linspace(0.0, angle_bp[-1], 1001)])
    linear = np.radians(np.interp(self.angle_grid, angle_bp, linear_bp))
    self.path_grid = linear / np.interp(self.angle_grid, table.ratio_bp, table.ratio_v)
    assert np.all(np.diff(self.path_grid) > 0), "rack map must be monotonic to invert"

  def _angle(self, path: float) -> float:
    if path > self.path_grid[-1]:
      return float(self.angle_grid[-1] * path / self.path_grid[-1])
    return float(np.interp(path, self.path_grid, self.angle_grid))

  def _path(self, angle_deg: float) -> float:
    if angle_deg > self.angle_grid[-1]:
      return float(self.path_grid[-1] * angle_deg / self.angle_grid[-1])
    return float(np.interp(angle_deg, self.angle_grid, self.path_grid))

  def angle_from_curvature(self, curvature: float, v_ego: float, roll: float) -> float:
    k = -curvature
    path = self.wheelbase * (k * (1.0 - self.slip_factor * v_ego ** 2) - GRAVITY * self.slip_factor * roll)
    return math.copysign(self._angle(abs(path)), path)

  def curvature_from_angle(self, angle_deg: float, v_ego: float, roll: float) -> float:
    path = math.copysign(self._path(abs(angle_deg)), angle_deg)
    k = (path / self.wheelbase + GRAVITY * self.slip_factor * roll) / (1.0 - self.slip_factor * v_ego ** 2)
    return -k
