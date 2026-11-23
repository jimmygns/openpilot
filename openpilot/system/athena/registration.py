#!/usr/bin/env python3
import hashlib
from pathlib import Path

from openpilot.common.api import get_key_pair
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.common.hardware import HARDWARE, PC
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog


UNREGISTERED_DONGLE_ID = "UnregisteredDevice"

def is_registered_device() -> bool:
  dongle = Params().get("DongleId")
  return dongle not in (None, UNREGISTERED_DONGLE_ID)


def register(show_spinner=False) -> str | None:
  """
  Offline-only device ID resolution for PTM builds (never contacts comma servers).
  Uses cached DongleId or /persist/comma/dongle_id when available, otherwise
  derives a deterministic local ID on device so startup never blocks or phones home.
  """
  del show_spinner
  params = Params()

  dongle_id: str | None = params.get("DongleId")
  if not dongle_id and Path(Paths.persist_root() + "/comma/dongle_id").is_file():
    with open(Paths.persist_root() + "/comma/dongle_id") as f:
      dongle_id = f.read().strip()

  _, _, public_key = get_key_pair()
  if not public_key:
    dongle_id = UNREGISTERED_DONGLE_ID
    cloudlog.warning("missing public key")
  elif not dongle_id or dongle_id == UNREGISTERED_DONGLE_ID:
    if PC:
      dongle_id = UNREGISTERED_DONGLE_ID
    else:
      serial = HARDWARE.get_serial()
      dongle_id = hashlib.sha256(f"local-{serial}".encode()).hexdigest()[:16]

  if dongle_id:
    params.put("DongleId", dongle_id, block=True)
    set_offroad_alert("Offroad_UnregisteredHardware", False)
  return dongle_id


if __name__ == "__main__":
  print(register())
