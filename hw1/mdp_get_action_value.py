def get_action_value(mdp, state_values, state, action, gamma):
    """ 
    Computes Q(s,a) as in formula above
    for all the proceeding states which we can go from the current
    we accumulate rewards + v_i which we'll count later (it's a state value in fact for all states)
    """

    next_states = mdp.get_next_states(action=action, state=state)
    q_i = 0
    for next_state in next_states:
        v_i = state_values[next_state]
        r = mdp.get_reward(state=state, action=action, next_state=next_state)
        p = mdp.get_transition_prob(action=action, next_state=next_state, state=state)
#         print(next_state, v_i, r, p)
        q_i = q_i + p * (r + gamma * v_i)
    return q_i
