"""Offline regression for HORA Stage 2 actor/history semantics; no SDK connection."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/revo3"))
from revo3_deploy.input_builder import Stage2InputBuilder
from revo3_deploy.robot_profile import Revo3Profile


class DeploymentInputsTest(unittest.TestCase):
    def test_contact_blind_actor_retains_adaptation_contacts(self):
        builder = Stage2InputBuilder(-np.ones(21), np.ones(21), [str(i) for i in range(21)])
        initial_contacts = np.array([1, 0, 0, 1, 0], dtype=np.float32)
        inputs = builder.reset(np.zeros(21), initial_contacts)
        self.assertEqual(inputs["obs"].shape, (1, 141))
        self.assertEqual(inputs["proprio_hist"].shape, (1, 30, 47))
        np.testing.assert_array_equal(inputs["obs"].reshape(3, 47)[:, -5:], 0)
        np.testing.assert_array_equal(inputs["proprio_hist"][0, -1, -5:], initial_contacts)
        next_contacts = 1 - initial_contacts
        inputs = builder.observe(np.full(21, 0.5), next_contacts)
        np.testing.assert_array_equal(inputs["proprio_hist"][0, -2, -5:], initial_contacts)
        np.testing.assert_array_equal(inputs["proprio_hist"][0, -1, -5:], next_contacts)
        np.testing.assert_array_equal(inputs["obs"].reshape(3, 47)[:, -5:], 0)
        np.testing.assert_allclose(inputs["obs"][0, -47:-26], 0.5)

    def test_new_profile_is_explicitly_uncalibrated(self):
        profile = Revo3Profile.load(Path(__file__).resolve().parents[1] / "deploy/revo3/config/revo3_right.yaml")
        self.assertFalse(profile.hardware_calibrated)
        self.assertEqual(len(profile.asset_revision), 40)
        builder = Stage2InputBuilder(profile.joint_lower_policy, profile.joint_upper_policy, profile.policy_joint_order)
        builder.reset(np.zeros(21), np.zeros(5), target=profile.joint_upper_policy)
        np.testing.assert_allclose(builder.action_to_target(np.ones(21)), profile.joint_upper_policy)
        with self.assertRaises(ValueError):
            builder.observe(np.zeros(21), np.zeros(4))


if __name__ == "__main__":
    unittest.main()
