# Copyright 2018 The YARL-Project, All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import unittest

from yarl.agents import DQNAgent
import yarl.spaces as spaces
from yarl.envs import GridWorld, RandomEnv
from yarl.execution.single_threaded_worker import SingleThreadedWorker
from yarl.utils import root_logger
from yarl.tests.agent_test import AgentTest


class TestDQNAgentFunctionality(unittest.TestCase):
    """
    Tests the DQN Agent's assembly and functionality.
    """
    root_logger.setLevel(level=logging.DEBUG)

    def test_dqn_assembly(self):
        """
        Creates a DQNAgent and runs it for a few steps in the random Env.
        """
        env = RandomEnv(state_space=spaces.IntBox(2), action_space=spaces.IntBox(2), deterministic=True)
        agent = DQNAgent.from_spec(
            "configs/dqn_agent_for_random_env.json",
            double_q=False,
            dueling_q=False,
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(environment=env, agent=agent)
        timesteps = 100
        results = worker.execute_timesteps(timesteps, use_exploration=False)

        print(results)

        self.assertEqual(results["timesteps_executed"], timesteps)
        self.assertEqual(results["env_frames"], timesteps)
        # Assert deterministic execution of Env and Agent.
        self.assertAlmostEqual(results["mean_episode_reward"], 5.923551400230593)
        self.assertAlmostEqual(results["max_episode_reward"], 14.312868008192979)
        self.assertAlmostEqual(results["final_episode_reward"], 0.14325251090518198)

    def test_dqn_functionality(self):
        """
        Creates a DQNAgent and runs it for a few steps in a GridWorld to vigorously test
        all steps of the learning process.
        """
        env = GridWorld(world="2x2", save_mode=True)  # no holes, just fire
        agent = DQNAgent.from_spec(  # type: DQNAgent
            "configs/dqn_agent_for_functionality_test.json",
            double_q=True,
            dueling_q=True,
            state_space=env.state_space,
            action_space=env.action_space,
            store_last_memory_batch=True
        )
        worker = SingleThreadedWorker(environment=env, agent=agent)
        test = AgentTest(worker=worker)

        # 1st step -> Expect insert into python-buffer.
        # action: up (0)
        test.step(1, reset=True)
        # Environment's new state.
        test.check_env("state", 0)
        # Agent's buffer.
        test.check_agent("states_buffer", [0])
        test.check_agent("actions_buffer", [0])
        test.check_agent("rewards_buffer", [-1.0])
        test.check_agent("terminals_buffer", [False])
        # Memory contents.
        test.check_var("replay-memory/index", 0)
        test.check_var("replay-memory/size", 0)
        test.check_var("replay-memory/memory/states", np.array([[0] * 4] * agent.memory.capacity))
        test.check_var("replay-memory/memory/actions", np.array([0] * agent.memory.capacity))
        test.check_var("replay-memory/memory/rewards", np.array([0] * agent.memory.capacity))
        test.check_var("replay-memory/memory/terminals", np.array([False] * agent.memory.capacity))
        # Check policy and target-policy weights (should be the same).
        test.check_var("policy/neural-network/hidden/dense/kernel", np.array([[0.9] * 2] * 4))
        test.check_var("target-policy/neural-network/hidden/dense/kernel", np.array([[0.9] * 2] * 4))
        test.check_var("policy/action-adapter/action-layer/dense/kernel", np.array([[0.8] * 5] * 2))
        test.check_var("target-policy/action-adapter/action-layer/dense/kernel", np.array([[0.8] * 5] * 2))

        # 2nd step -> expect insert into memory (and python buffer should be empty again).
        # action: up (0)
        # Also check the policy and target policy values (Should be equal at this point).
        test.step(1)
        test.check_env("state", 0)
        test.check_agent("states_buffer", [])
        test.check_agent("actions_buffer", [])
        test.check_agent("rewards_buffer", [])
        test.check_agent("terminals_buffer", [])
        test.check_var("replay-memory/index", 2)
        test.check_var("replay-memory/size", 2)
        test.check_var("replay-memory/memory/states", np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]] +
                                                               [[0.0, 0.0, 0.0, 0.0]] * (agent.memory.capacity - 2)))
        test.check_var("replay-memory/memory/actions", np.array([0, 0] + [0] * (agent.memory.capacity - 2)))
        test.check_var("replay-memory/memory/rewards", np.array([-1.0, -1.0] + [0.0] * (agent.memory.capacity - 2)))
        test.check_var("replay-memory/memory/terminals", np.array([False] * agent.memory.capacity))
        # Check policy and target-policy weights (should be the same).
        test.check_var("policy/neural-network/hidden/dense/kernel", np.array([[0.9] * 2] * 4))
        test.check_var("target-policy/neural-network/hidden/dense/kernel", np.array([[0.9] * 2] * 4))
        test.check_var("policy/action-adapter/action-layer/dense/kernel", np.array([[0.8] * 5] * 2))
        test.check_var("target-policy/action-adapter/action-layer/dense/kernel", np.array([[0.8] * 5] * 2))

        # 3rd and 4th step -> expect another insert into memory (and python buffer should be empty again).
        # actions: down, left
        # Expect an update to the policy variables (leave target as is (no sync yet)).
        test.step(2)
        test.check_env("state", 0)
        test.check_agent("states_buffer", [])
        test.check_agent("actions_buffer", [])
        test.check_agent("rewards_buffer", [])
        test.check_agent("terminals_buffer", [])
        test.check_var("replay-memory/index", 4)
        test.check_var("replay-memory/size", 4)
        test.check_var("replay-memory/memory/states", np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                                                [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]] +
                                                               [[0.0, 0.0, 0.0, 0.0]] * (agent.memory.capacity - 4)))
        test.check_var("replay-memory/memory/actions", np.array([0, 0, 0, 0] + [0] * (agent.memory.capacity - 4)))
        test.check_var("replay-memory/memory/rewards", np.array([-1.0, -1.0, -1.0, -1.0] + [0.0] * (agent.memory.capacity - 4)))
        test.check_var("replay-memory/memory/terminals", np.array([False] * agent.memory.capacity))
        # Get the latest memory batch.
        expected_batch = dict(
            states=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            actions=np.array([0, 0]),
            rewards=np.array([-1.0, -1.0]),
            terminals=np.array([False, False]),
            next_states=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        )
        test.check_agent("last_memory_batch", expected_batch)
        matrix1_qnet = np.array([[0.9] * 2] * 4)
        matrix2_qnet = np.array([[0.8] * 2] * 4)
        matrix1_target_net = np.array([[0.9] * 2] * 4)
        matrix2_target_net = np.array([[0.8] * 2] * 4)
        # Check policy and target-policy weights (policy should be updated now).
        test.check_var("policy/neural-network/hidden/dense/kernel", matrix1_qnet)
        test.check_var("target-policy/neural-network/hidden/dense/kernel", matrix1_target_net)
        test.check_var("policy/action-adapter/action-layer/dense/kernel", matrix2_qnet)
        test.check_var("target-policy/action-adapter/action-layer/dense/kernel", matrix2_target_net)
        # Calculate gradient per weight based on the above batch.
        #for

    def _helper_get_q_values(self, input_, matrix1, matrix2):
        """
        Calculates the q-values for a given simple 1-hidden 1-action-layer (both linear w/o biases) setup.

        Args:
            input_ (np.ndarray): The input array (batch x in-nodes).
            matrix1 (np.ndarray): The weights matrix of the hidden layer.
            matrix2 (np.ndarray): The weights matrix of the action-layer.

        Returns:
            np.ndarray: The calculated q-values.
        """
        nn_output = np.matmul(np.matmul(input_, matrix1), matrix2)
        #q_values =
