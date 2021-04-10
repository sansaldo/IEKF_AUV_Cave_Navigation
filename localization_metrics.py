import numpy as np
from sklearn.metrics import mean_squared_error

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
        print(gt_new)

        predicted_new = []
        for t, sample in zip(predicted_times, predicted):
            if t in gt_times:
                predicted_new.append(sample)
        predicted_new = np.array(predicted_new)
        print(predicted_new)

        mse = mean_squared_error(gt_new.flatten(), predicted_new.flatten())

    return mse