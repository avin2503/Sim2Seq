import numpy as np
import pandas as pd

def extract_event_data(data_df, events_df_row):
    event_data = data_df.iloc[
        events_df_row["start_idx"]:
        events_df_row["end_idx"]+1]
    return event_data

def cpic_greaterthan1000(x):
    a = 2.456
    b = 1.187
    c = 2.73
    return a * np.log(np.log(x)) + b * np.log(np.log(np.log(x))) + c

def cpic_lessthan1000(x):
    a = 1.239
    b = 0.9872
    c = 1.999
    p3 = 5.913e-10
    p4 = -1.876e-06
    p5 = 0.004354
    ph = -0.1906
    return a * np.log(np.log(x)) + b * np.log(x) + p3 * x**3 + p4 * x**2 + p5 * x + ph * np.abs(x)**0.5 + c

def stepFinder_CPIC(data, sensitivity=1, min_level_length=2):
    original_data_mapping = np.arange(len(data))
    data_length = len(data)
    original_data_mapping = original_data_mapping[~np.isnan(data)]
    data = data[~np.isnan(data)]

    cpic_multiplier = sensitivity
    min_level_length = min_level_length

    x = np.concatenate(([0], np.cumsum(data)))
    xsq = np.concatenate(([0], np.cumsum(data**2)))

    def find_transitions(left, right):
        transitions = []
        N_T = right - left + 1
        N_L = np.arange(min_level_length, N_T - min_level_length + 1)
        N_R = N_T - N_L

        if N_T < 2 * min_level_length:
            return transitions

        x_mean_L = (x[left + N_L - 1] - x[left - 1]) / N_L
        x_mean_R = (x[right] - x[right - N_R]) / N_R
        x_mean_T = (x[right] - x[left - 1]) / N_T

        xsq_mean_L = (xsq[left + N_L - 1] - xsq[left - 1]) / N_L
        xsq_mean_R = (xsq[right] - xsq[right - N_R]) / N_R
        xsq_mean_T = (xsq[right] - xsq[left - 1]) / N_T

        var_L = np.maximum(xsq_mean_L - x_mean_L**2, 0.0003)
        var_R = np.maximum(xsq_mean_R - x_mean_R**2, 0.0003)
        var_T = np.maximum(xsq_mean_T - x_mean_T**2, 0.0003)

        if N_T >= 1e6:
            p_CPIC = cpic_greaterthan1000(1e6)
        elif N_T > 1000 and N_T < 1e6:
            p_CPIC = cpic_greaterthan1000(N_T)
        else:
            p_CPIC = cpic_lessthan1000(N_T)

        CPIC = 0.5 * (N_L * np.log(var_L) + N_R * np.log(var_R) - N_T * np.log(var_T)) + 1 + cpic_multiplier * p_CPIC

        min_CPIC_index = np.argmin(CPIC)
        min_index = min_CPIC_index + min_level_length + left - 1

        if CPIC[min_CPIC_index] < 0:
            transitions = [min_index] + find_transitions(left, min_index) + find_transitions(min_index + 1, right)

        return transitions

    transitions = sorted(find_transitions(1, len(data)))
    transitions = [t for t in transitions if t > min_level_length and t < len(data) - min_level_length]
    transitions_with_ends = [0] + transitions + [len(data)]

    some_change = True
    while some_change:
        some_change = False
        transition_CPIC = -np.inf * np.ones(len(transitions_with_ends))

        for ii in range(1, len(transitions_with_ends) - 1):
            left = max(transitions_with_ends[ii - 1], 1)
            right = transitions_with_ends[ii + 1]

            N_T = right - left + 1
            N_L = transitions_with_ends[ii] - left + 1
            N_R = right - transitions_with_ends[ii]

            if N_T < 2 * min_level_length:
                continue

            x_mean_L = (x[left + N_L - 1] - x[left - 1]) / N_L
            x_mean_R = (x[right] - x[right - N_R]) / N_R
            x_mean_T = (x[right] - x[left - 1]) / N_T

            xsq_mean_L = (xsq[left + N_L - 1] - xsq[left - 1]) / N_L
            xsq_mean_R = (xsq[right] - xsq[right - N_R]) / N_R
            xsq_mean_T = (xsq[right] - xsq[left - 1]) / N_T

            var_L = np.maximum(xsq_mean_L - x_mean_L**2, 0.0003)
            var_R = np.maximum(xsq_mean_R - x_mean_R**2, 0.0003)
            var_T = np.maximum(xsq_mean_T - x_mean_T**2, 0.0003)

            if N_T >= 1e6:
                p_CPIC = cpic_greaterthan1000(1e6)
            elif N_T > 1000 and N_T < 1e6:
                p_CPIC = cpic_greaterthan1000(N_T)
            else:
                p_CPIC = cpic_lessthan1000(N_T)

            transition_CPIC[ii] = 0.5 * (N_L * np.log(var_L) + N_R * np.log(var_R) - N_T * np.log(var_T)) + 1 + cpic_multiplier * p_CPIC

        max_CPIC_index = np.argmax(transition_CPIC)
        if transition_CPIC[max_CPIC_index] > 0:
            transitions_with_ends.pop(max_CPIC_index)
            some_change = True

    features = np.zeros((2, len(transitions_with_ends) - 1))
    errors = np.zeros((2, len(transitions_with_ends) - 1))
    stiffnesses = [None] * (len(transitions_with_ends) - 1)

    for ct in range(1, len(transitions_with_ends)):
        features[:, ct-1] = [np.median(data[transitions_with_ends[ct-1]:transitions_with_ends[ct]]),
                             np.std(data[transitions_with_ends[ct-1]:transitions_with_ends[ct]])]

        errors[:, ct-1] = [features[1, ct-1] / np.sqrt(transitions_with_ends[ct] - transitions_with_ends[ct-1] - 1),
                           features[1, ct-1] / np.sqrt(2 * (transitions_with_ends[ct] - transitions_with_ends[ct-1] - 1))]

        stiffnesses[ct-1] = np.diag(errors[:, ct-1]**-2)

    transitions = [0] + [original_data_mapping[t] for t in transitions_with_ends[1:-1]] + [data_length]

    return np.array(transitions), np.array(features), np.array(errors), np.array(stiffnesses)

def format_steps_df(event_meta, transitions, features, errors, stiffnesses):
    samplingFreq =  event_meta["n_pts"] / event_meta["duration(s)"]

    steps_df = pd.DataFrame({
            "step_start_idx": transitions[:-1],
            "step_end_idx": transitions[1:],
        })
    steps_df["n_pts"] = steps_df["step_end_idx"] - steps_df["step_start_idx"]
    steps_df["dwell_time(s)"] = steps_df["n_pts"] / samplingFreq
    steps_df["raw_mean"] = features[0, :]
    steps_df["raw_std"] = features[1, :]
    
    return steps_df

    
