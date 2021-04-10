import numpy as np
from sklearn.metrics import mean_squared_error
from itertools import product
from constants import cone_times, cone_distances

# Calculates MSE of ground truth (gt) and predicted trajectories
# gt and predicted should be shape (number of timesteps, number of dimensions),
# or if different number of timesteps, pass in times objects for each of shape (number of timesteps)
def mse(gt, predicted, gt_times=None, predicted_times=None):
    gt = np.array(gt)
    predicted = np.array(predicted)
    if gt_times is not None:
        gt_times = np.array(gt_times)
    if predicted_times is not None:
        predicted_times = np.array(predicted_times)

    assert gt.shape[0] == predicted.shape[0] or (gt_times is not None and predicted_times is not None), "Ground truth and predicted trajectories should have same number of values. If they don't, pass their times into this function so we can match them up."
    assert gt.shape[1] == predicted.shape[1], "Comparison results must have same number of dimensions."

    if gt.shape[0] == predicted.shape[0]:
        mse = mean_squared_error(gt.flatten(), predicted.flatten())
    else:
        # Only evaluate where the times overlap
        gt_new = []
        for t, sample in zip(gt_times, gt):
            if t in predicted_times:
                gt_new.append(sample)
        gt_new = np.array(gt_new)

        predicted_new = []
        for t, sample in zip(predicted_times, predicted):
            if t in gt_times:
                predicted_new.append(sample)
        predicted_new = np.array(predicted_new)

        mse = mean_squared_error(gt_new.flatten(), predicted_new.flatten())

    return mse

# Measures 2 cone metrics from Angelos et al., 2016
# pos: array of shape (number of pred timesteps, 3) for xyz predictions to evaluate
# times: array of shape (number of pred timesteps) for the evaluated estimation
# cone_offsets: (optional) array of shape (number of cones, 2, 3) containing the xyz offset for each cone based on first and second camera observation
# Returns a dictionary of metrics
def cone_metrics(pos, times, cone_offsets=None):
    return_metrics = {}
    no_cones = cone_times.shape[0]
    for i in range(no_cones):
        cone_time0 = cone_times[i, 0]
        cone_time1 = cone_times[i, 1]
        
        pred_xyz0 = pos[np.argmin(np.abs(times - cone_time0)), :]
        pred_xyz1 = pos[np.argmin(np.abs(times - cone_time1)), :]

        if cone_offsets is not None:
            pred_xyz0 += cone_offsets[i,0,:]
            pred_xyz1 += cone_offsets[i,1,:]

        return_metrics['%s_2pass_abs_error' % str(i)] = np.abs(pred_xyz0 - pred_xyz1)
        return_metrics['%s_2pass_error^2' % str(i)] = (pred_xyz0 - pred_xyz1) ** 2
        return_metrics['%s_2pass_2norm' % str(i)] = np.linalg.norm(pred_xyz0 - pred_xyz1, ord=2)

    for i, j in product(list(range(no_cones)), repeat=2):
        # We don't have all distances - if we don't, don't compute metric
        if cone_distances[i,j] <= 0.:
            continue

        # First cone (first and second pass)
        cone0_time0 = cone_times[i, 0]
        cone0_time1 = cone_times[i, 1]
        pred0_xyz0 = pos[np.argmin(times - cone0_time0), :]
        pred0_xyz1 = pos[np.argmin(times - cone0_time1), :]

        # Second cone (first and second pass)
        cone1_time0 = cone_times[j, 0]
        cone1_time1 = cone_times[j, 1]
        pred1_xyz0 = pos[np.argmin(np.abs(times - cone1_time0)), :]
        pred1_xyz1 = pos[np.argmin(np.abs(times - cone1_time1)), :]

        if cone_offsets is not None:
            pred0_xyz0 += cone_offsets[i,0,:]
            pred0_xyz1 += cone_offsets[i,1,:]

            pred1_xyz0 += cone_offsets[j,0,:]
            pred1_xyz1 += cone_offsets[j,1,:]            

        # Distance of first pass between these cones, and second pass between these cones
        dist0 = np.linalg.norm(pred0_xyz0 - pred1_xyz0)
        dist1 = np.linalg.norm(pred0_xyz1 - pred1_xyz1)

        gt_cone_distance = cone_distances[i,j]
        error0 = dist0 - gt_cone_distance
        error1 = dist1 - gt_cone_distance

        # Add each pass' error and the average pass error
        return_metrics['%s_%s_pass0_error' % (str(i), str(j))] = error0
        return_metrics['%s_%s_pass1_error' % (str(i), str(j))] = error1
        return_metrics['%s_%s_avg_error' % (str(i), str(j))] = (error0 + error1) / 2.

    return return_metrics
        


    

