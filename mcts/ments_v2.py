class _Node:
    def __init__(self, parent: Optional["_Node"], action: Optional[int], env: SimpleEnv):
        self.parent = parent
        self.action = action
        self.env = env.clone()
        self.children: Dict[int, '_Node'] = {}
        self.valid_actions = list(env.legal_actions)
        self.untried_actions = list(env.legal_actions)
        self.visits = 0
        self.value = 0.0
        self.Qsft = 0