from rlgraph.components import Component
from rlgraph.agents import Agent
from rlgraph.components.memories import Memory
from rlgraph.spaces import FloatBox, BoolBox


class MBPOWrapperComponent(Component):
    pass


class MBPOWrapper(Agent):
    def __init__(
            self,
            agent,
            preprocessing_spec=None,
            exploration_spec=None,
            execution_spec=None,
            optimizer_spec=None,
            observe_spec=None,
            update_spec=None,
            summary_spec=None,
            saver_spec=None,
            auto_build=True,
            name="mbpo-wrapper",
            memory_spec=None
    ):
        self.agent = agent
        super().__init__(
            state_space=agent.state_space,
            action_space=agent.action_space,
            discount=agent.discount,
            preprocessing_spec=preprocessing_spec,
            network_spec=None,
            internal_states_space=None,
            policy_spec=None,
            value_function_spec=None,
            exploration_spec=exploration_spec,
            execution_spec=execution_spec,
            optimizer_spec=optimizer_spec,
            value_function_optimizer_spec=None,
            observe_spec=observe_spec,
            update_spec=update_spec,
            summary_spec=summary_spec,
            saver_spec=saver_spec,
            auto_build=auto_build,
            name=name
        )

        # Extend input Space definitions to this Agent's specific API-methods.
        preprocessed_state_space = self.preprocessed_state_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)
        float_action_space = self.action_space.with_batch_rank().map(
            mapping=lambda flat_key, space: space.as_one_hot_float_space() if isinstance(space, IntBox) else space
        )

        self.memory = Memory.from_spec(memory_spec)

    def get_action(self, states, internals=None, use_exploration=True, apply_preprocessing=True, extra_returns=None,
                   time_percentage=None):
        """
        Delegates the action to the wrapped agent
        :param states:
        :param internals:
        :param use_exploration:
        :param apply_preprocessing:
        :param extra_returns:
        :param time_percentage:
        :return:
        """
        return self.agent.get_action(
            states=states,
            internals=internals,
            use_exploration=use_exploration,
            apply_preprocessing=apply_preprocessing,
            extra_returns=extra_returns,
            time_percentage=time_percentage
        )

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, next_states, terminals):
        # Stash in the buffer
        pass

    def update(self, batch=None, time_percentage=None, **kwargs):
        # 1. Train the model
        # 2. Sample trajectories
        # 3. Update the wrapped agent
        pass

    def __repr__(self):
        return "{}(agent={})".format(self.__class__.__name__, repr(self.agent))
