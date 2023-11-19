
import sys
import numpy as np
import matplotlib.pyplot as plt

def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

args = sys.argv

CM_SCOREFILE = 'RawGAT_ST_cat_LA_eval_CM_scores_RR_mod1.txt'
     
# Replace CM scores with your own scores or provide score file as the first argument.
cm_scores_file =  CM_SCOREFILE

# Load CM scores
cm_data = np.genfromtxt(cm_scores_file, dtype=str)
cm_utt_id = cm_data[:, 0]
cm_sources = cm_data[:, 1]
cm_keys = cm_data[:, 2]
cm_scores = cm_data[:, 3].astype(float)

# Extract bona fide (real bonafide) and spoof scores from the CM scores
bona_cm = cm_scores[cm_keys == 'bonafide']
spoof_cm = cm_scores[cm_keys == 'spoof']

# EERs of the standalone systems and fix ASV operating point to EER threshold
eer_cm = compute_eer(bona_cm, spoof_cm)[0]

print('\nCM SYSTEM')
print('   EER            = {:8.9f} % (Equal error rate for countermeasure)'.format(eer_cm * 100))