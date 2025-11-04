import util, random

class Agent:

  def getAction(self, state):
    """
    For the given state, get the agent's chosen
    action.  The agent knows the legal actions
    """
    abstract

  def getValue(self, state):
    """
    Get the value of the state.
    """
    abstract

  def getQValue(self, state, action):
    """
    Get the q-value of the state action pair.
    """
    abstract

  def getPolicy(self, state):
    """
    Get the policy recommendation for the state.

    May or may not be the same as "getAction".
    """
    abstract

  def update(self, state, action, nextState, reward):
    """
    Update the internal state of a learning agent
    according to the (state, action, nextState)
    transistion and the given reward.
    """
    abstract


class RandomAgent(Agent):
  """
  Clueless random agent, used only for testing.
  """

  def __init__(self, actionFunction):
    self.actionFunction = actionFunction

  def getAction(self, state):
    return random.choice(self.actionFunction(state))

  def getValue(self, state):
    return 0.0

  def getQValue(self, state, action):
    return 0.0

  def getPolicy(self, state):
    return 'random'

  def update(self, state, action, nextState, reward):
    pass


################################################################################
# Exercise 2

class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp= mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        self.V = {s: 0 for s in states}  #initialize all state values to 0

        for i in range(iterations):
            newV = self.V.copy()  #copy for synchronous updates

            for state in states:
                if self.mdp.isTerminal(state):
                    newV[state] = 0
                    continue

                newV[state] = max(
                    sum(prob*(self.mdp.getReward(state, action, next_state) + discount * self.V[next_state])
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(
                            state, action
                        )
                    )
                    for action in self.mdp.getPossibleActions(state)
                )

            self.V = newV  #update after going through all states


    def getValue(self, state):
        """Return the value of the state."""
        return self.V[state]


    def getQValue(self, state, action):
        """Return the Q-value of the (state, action) pair."""
        q_pi_s_a = sum(
            prob
            * (
                self.mdp.getReward(state, action, next_state)
                + self.discount * self.V[next_state]
            )
            for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)
        )
        return q_pi_s_a  

    def getPolicy(self, state):
        """Return the best action according to the computed policy."""
        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)
        best_action = None
        best_q_val = float("-inf")

        for action in actions:
            q_pi_s_a = self.getQValue(state, action)
            if q_pi_s_a > best_q_val:
                best_q_val = q_pi_s_a
                best_action = action

        return best_action


    def getAction(self, state):
        """Return the action recommended by the policy."""
        return self.getPolicy(state)


    def update(self, state, action, nextState, reward):
        """Not used for value iteration agents."""
        pass


################################################################################
# Exercise 3

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=1):
        """
        Your policy iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()

        # 1. Initialization
        self.V = {s: 0 for s in states}  # initialize all state values to 0
        self.policy = {}
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                self.policy[state] = actions[0] if actions else None
            else:
                self.policy[state] = None

        # 2. Policy Iteration Loop
        roundcounts = 0
        while self.iterations > 0:
            roundcounts += 1
            # 2.1. Policy Evaluation
            for i in range(100):  # hard coded to 100
                newV = self.V.copy()  #copy for synchronous updates

                for state in states:
                    if self.mdp.isTerminal(state):
                        newV[state] = 0
                        continue

                    action = self.policy[state]
                    if action is not None:
                        newV[state] = sum(
                            prob * (self.mdp.getReward(state, action, next_state)
                                    + discount * self.V[next_state])
                            for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)
                        )

                self.V = newV

            # 2.2. Policy Improvement
            policy_stable = True
            for state in states:
                if self.mdp.isTerminal(state):
                    continue

                old_action = self.policy[state]

                # Find best action based on current value function
                best_action = None
                best_value = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    q_value = sum(
                        prob * (self.mdp.getReward(state, action, next_state)
                                + discount * self.V[next_state])
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)
                    )
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action

                self.policy[state] = best_action

                # Check if policy changed
                if old_action != best_action:
                    policy_stable = False

            # If policy didn't change, we've converged
            if policy_stable:
                print(f"Converged after {roundcounts} rounds.")
                break

            if roundcounts >= self.iterations:
                print(f"Reached specified policy iteration loops of {self.iterations}.")
                break


    def getValue(self, state):
        """Return the value of the state."""
        return self.V[state]

    def getQValue(self, state, action):
        """Return the Q-value of the (state, action) pair."""
        q_pi_s_a = sum(
            prob * (self.mdp.getReward(state, action, next_state)
                    + self.discount * self.V[next_state])
            for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)
        )
        return q_pi_s_a

    def getPolicy(self, state):
        """Return the best action according to the computed policy."""
        return self.policy.get(state, None)

    def getAction(self, state):
        """Return the action recommended by the policy."""
        return self.getPolicy(state)

################################################################################
# Below can be ignored for Exercise 2

class QLearningAgent(Agent):

  def __init__(self, actionFunction, discount = 0.9, learningRate = 0.1, epsilon = 0.2):
    """
    A Q-Learning agent gets nothing about the mdp on
    construction other than a function mapping states to actions.
    The other parameters govern its exploration
    strategy and learning rate.
    """
    self.setLearningRate(learningRate)
    self.setEpsilon(epsilon)
    self.setDiscount(discount)
    self.actionFunction = actionFunction

    raise "Your code here."




  # THESE NEXT METHODS ARE NEEDED TO WIRE YOUR AGENT UP TO THE CRAWLER GUI

  def setLearningRate(self, learningRate):
    self.learningRate = learningRate

  def setEpsilon(self, epsilon):
    self.epsilon = epsilon

  def setDiscount(self, discount):
    self.discount = discount

  # GENERAL RL AGENT METHODS

  def getValue(self, state):
    """
    Look up the current value of the state.
    """

    raise ValueError("Your code here.")



  def getQValue(self, state, action):
    """
    Look up the current q-value of the state action pair.
    """

    raise ValueError("Your code here.")



  def getPolicy(self, state):
    """
    Look up the current recommendation for the state.
    """

    raise ValueError("Your code here.")



  def getAction(self, state):
    """
    Choose an action: this will require that your agent balance
    exploration and exploitation as appropriate.
    """

    raise ValueError("Your code here.")



  def update(self, state, action, nextState, reward):
    """
    Update parameters in response to the observed transition.
    """

    raise ValueError("Your code here.")
