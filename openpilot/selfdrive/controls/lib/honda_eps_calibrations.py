"""EPS firmware calibration for 2020+ Honda CR-V 5G (39990-TLA,A220 PTM rev b, RWD sha256 2c992997)."""

from opendbc.car.honda.steer_ratio import HONDA_VGR_CRV_TLA_A040
from openpilot.selfdrive.controls.lib.honda_eps_firmware_ff import (
  ColumnLoadModel,
  EpsFirmwareCalibration,
  R6_GAIN_BP,
  computed_r6,
)
from openpilot.selfdrive.controls.lib.honda_eps_rack_map import CRV_TLA_A040_RACK

# CR-V 5G column load (fitted on 40,193 hands-off active frames, R^2 = 0.843, 60 s holdouts 0.857 / 0.820)
CRV_TLA_A040_LOAD = ColumnLoadModel(
  k0=-7.29446, k1=-0.143159, c=-4.60337, friction=-297.83, bias=-19.899, kroll=-3.58048
)

CRV_TLA_A040_P_SCALE = (1.0, 1.0, 1.0)
CRV_TLA_A040_I_SCALE = (1.0, 1.0, 1.0)

P_ROW_PMINUS5 = (117, 148, 184, 220, 245, 257, 263, 265, 265)
CLARITY_TRW_A020_P_KEYS = (0, 222, 443, 665, 887, 1108, 1330, 1552, 1774)

CRV_TLA_A040_R6_CENTER, _, CRV_TLA_A040_R6_GAIN = computed_r6(HONDA_VGR_CRV_TLA_A040)

# CR-V 5G 39990-TLA-A220 (2020+), corrected PTM build of 2026-10-16 rev b (payload a62337a9, RWD 2c992997):
# Clarity-equivalent command map & P table on A220's axes, D737, KFF45, Norm1650, Trk3200, key clamp 1774,
# and road-measured CRV_TLA_A040_RACK validated on CR-V A220 VSA yaw (routes 030a, 030d, 0039, 003d).
CRV_TLA_A220 = EpsFirmwareCalibration(
  name="crv_tla_a220",
  e4_per_output=4096.0,
  r5_key_bp=(0, 161, 222, 322, 409, 534, 696, 998, 1663),
  r5_v=(0, 2720, 3750, 5439, 6917, 9038, 11775, 16880, 28127),
  key_clamp=1774,
  envelope_bp=(0, 50, 100, 150, 200, 250, 300, 350, 400),
  envelope_v=(1774, 1774, 1774, 1774, 1774, 1774, 1330, 1330, 1330),
  kp_key_bp=tuple(round(k * 4096 / 3840) for k in CLARITY_TRW_A020_P_KEYS),
  kp_v=P_ROW_PMINUS5,
  r6_per_deg_s=CRV_TLA_A040_R6_CENTER,
  r6_angle_bp=R6_GAIN_BP,
  r6_angle_gain=CRV_TLA_A040_R6_GAIN,
  load=CRV_TLA_A040_LOAD,
  p_scale=CRV_TLA_A040_P_SCALE,
  i_scale=CRV_TLA_A040_I_SCALE,
  rack=CRV_TLA_A040_RACK,
)

REQUIRED_PTM_RWD = {
  "crv_tla_a220": "2c992997",
}

CALIBRATIONS_BY_CAR_AND_FW = {
  ("HONDA_CRV_5G", "39990-TLA-A220"): CRV_TLA_A220,
}


def select_eps_firmware_calibration(car_fingerprint: str, normalized_eps_fw: str) -> EpsFirmwareCalibration | None:
  return CALIBRATIONS_BY_CAR_AND_FW.get((car_fingerprint, normalized_eps_fw))
