def floating_mean(data, index, window):
    if index > window:
        floating_mean = np.zeros_like(data[:,index])
        for i in range(window):
            floating_mean += data[:,(index-window+1+i)]
        floating_mean = floating_mean/window
    else:
        floating_mean = data[:,index]
    return floating_mean