import hashlib
from Crypto.PublicKey import RSA
from pathlib import Path

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.params import Params
from openpilot.system.athena import registration
from openpilot.system.athena.registration import register, UNREGISTERED_DONGLE_ID
from openpilot.common.hardware.hw import Paths


class TestRegistration(OpenpilotTestCase):

  def setup_method(self):
    # clear params and setup key paths
    self.params = Params()

    persist_dir = Path(Paths.persist_root()) / "comma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    self.priv_key = persist_dir / "id_rsa"
    self.pub_key = persist_dir / "id_rsa.pub"
    self.dongle_id = persist_dir / "dongle_id"

  def _generate_keys(self):
    self.pub_key.touch()
    k = RSA.generate(2048)
    with open(self.priv_key, "wb") as f:
      f.write(k.export_key())
    with open(self.pub_key, "wb") as f:
      f.write(k.publickey().export_key())

  def test_valid_cache(self):
    # if all params are written, return the cached dongle id.
    # should work with a dongle ID on either /persist/ or normal params
    self._generate_keys()

    dongle = "DONGLE_ID_123"
    for persist, params in [(True, True), (True, False), (False, True)]:
      self.params.put("DongleId", dongle if params else "", block=True)
      with open(self.dongle_id, "w") as f:
        f.write(dongle if persist else "")
      assert register() == dongle

  def test_no_keys(self):
    # missing pubkey
    dongle = register()
    assert dongle == UNREGISTERED_DONGLE_ID
    assert self.params.get("DongleId") == dongle

  def test_missing_cache_pc_offline(self, mocker):
    self._generate_keys()
    mocker.patch.object(registration, "PC", True)
    dongle = register()
    assert dongle == UNREGISTERED_DONGLE_ID
    assert self.params.get("DongleId") == dongle

  def test_missing_cache_device_offline(self, mocker):
    self._generate_keys()
    mocker.patch.object(registration, "PC", False)
    mocker.patch.object(registration.HARDWARE, "get_serial", return_value="c3x-serial-001")
    expected = hashlib.sha256(b"local-c3x-serial-001").hexdigest()[:16]
    dongle = register()
    assert dongle == expected
    assert self.params.get("DongleId") == expected
