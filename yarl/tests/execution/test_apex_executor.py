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

import json
import os
import unittest
from yarl.execution.ray import ApexExecutor


class TestApexExecutor(unittest.TestCase):
    """
    Tests the ApexExecutor which provides an interface for distributing Apex-style workloads
    via Ray.
    """
    env_spec = dict(
      type="openai",
      gym_env="CartPole-v0"
    )

    cluster_spec = dict(
        redis_address=None,
        num_cpus=4,
        num_gpus=0,
        weight_sync_steps=64,
        replay_sampling_task_depth=1,
        env_interaction_task_depth=1,
        num_worker_samples=200,
        learn_queue_size=1,
        num_local_workers=1,
        num_remote_workers=1
    )

    def test_learning_cartpole(self):
        """
        Tests if apex can learn a simple environment using a single worker, thus replicating
        dqn.
        """
        path = os.path.join(os.getcwd(), "configs/apex_agent.json")
        with open(path, 'rt') as fp:
            agent_config = json.load(fp)

        # Cartpole settings from cartpole dqn test.
        agent_config.update(
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=64),
            optimizer_spec=dict(learning_rate=0.0002, clip_grad_norm=40.0)
        )

        # Define executor, test assembly.
        executor = ApexExecutor(
            environment_spec=self.env_spec,
            agent_config=agent_config,
            cluster_spec=self.cluster_spec
        )
        print("Successfully created executor.")

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(num_timesteps=5000))
        print("Finished executing workload:")
        print(result)
