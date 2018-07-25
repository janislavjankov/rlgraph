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

from yarl.agents.agent import Agent
from yarl.agents.dqn_agent import DQNAgent
from yarl.agents.apex_agent import ApexAgent
from yarl.agents.ppo_agent import PPOAgent
from yarl.agents.random_agent import RandomAgent


Agent.__lookup_classes__ = dict(
    dqn=DQNAgent,
    dqnagent=DQNAgent,
    apex=ApexAgent,
    apexagent=ApexAgent,
    ppo=PPOAgent,
    ppoagent=PPOAgent,
    random=RandomAgent
)

__all__ = ["Agent", "DQNAgent", "ApexAgent", "PPOAgent", "RandomAgent"]
