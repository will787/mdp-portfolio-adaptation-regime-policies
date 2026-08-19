import numpy as np


def belmann_equation(states, actions, transactions, rewards, discount_factor=0.99, theta=1e-5):
    """
      V(s): The value state s, which represents the long-term reward of being in state.
      R(s,a): The immediate reward received for taking action a in state s.
      Y: factor discount (between 0 and 1) that determines importance of future rewards.
      P(s'|s): The probability of transaction to state s' from state s (Independente da ação).
    """

    v = {s: 0 for s in states}
    policy = {s: None for s in states}

    while True:
        delta = 0
        for s in states:
            v_old = v[s]

            expected_future = sum(transactions.get((s, next_states), 0.0) * v[next_states] for next_states in states)
            immediate_reward = discount_factor * expected_future
    
            max_q_value = float('-inf')
            best_action = None

            for a in actions:
                #R(s,a)
                immediate_reward = rewards.get((s, a), 0.0)
                q_value = immediate_reward + (discount_factor * expected_future)

                #Q(s,a) = R(s,a) + gamma * Equação futura
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = a

            v[s] = max_q_value
            policy[s] = best_action
            delta = max(delta, abs(v_old - v[s])) 

        if delta < theta:
            break
    return v, policy    