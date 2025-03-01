from mlwpy import *
# 线形回归的目的是找到一条靠近所有点的直线

# 水平直线y=c
def axis_helper(ax, lims):
    ax.set_xlim(lims); ax.set_xticks([])
    ax.set_ylim(lims); ax.set_yticks([])

D = np.array([[3, 5],[4, 2]])

x, y = D[:, 0], D[:, 1]

horizontal_lines = np.array([1, 2, 3, 3.5, 4, 5])

results = []
fig, axes = plt.subplots(1, 6, figsize=(10, 5))
for h_line, ax in zip(horizontal_lines, axes.flat):
    axis_helper(ax, (0, 6))
    ax.set_title(str(h_line))

    ax.plot(x, y, 'ro')
    ax.axhline(h_line, color='y')

    predictions = h_line
    ax.vlines(x, predictions, y)

    errors = y - predictions
    sse = np.dot(errors, errors)

    results.append((predictions,
    errors, errors.sum(),
    sse,
    np.sqrt(sse)))

col_labels = "Prediction", "Errors", "Sum", "SSE", "Distance"
display(pd.DataFrame.from_records(results, columns=col_labels, index = "Prediction"))

# 倾斜直线y=mx+b
def process(D, model, ax):
    x, y = D[:, 0], D[:, 1]
    m, b = model

    axis_helper(ax, (0, 8))

    ax.plot(x, y, 'ro')

    helper_xs = np.array([0, 8])
    helper_line = m *helper_xs + b
    ax.plot(helper_xs, helper_line, color='y')

    predictions = m * x + b
    ax.vlines(x, predictions, y)

    errors = y - predictions

    sse = np.dot(errors, errors)
    return(errors, errors.sum(), sse, np.sqrt(sse))

D = np.array([[3, 5], [4, 2]])

lines_mb = np.array([[1, 0], [1, 1], [1, 2], [-1, 8], [-3, 14]])

col_labels = ("Raw Errors", "Sum", "SSE", "TotDist")
results = []

fig, axes = plt.subplots(1, 5, figsize=(12, 6))
records = [process(D, mod, ax) for mod, ax in zip(lines_mb, axes.flat)]
df = pd.DataFrame.from_records(records, columns=col_labels)
display(df)

# predicted = m*x + b
# error = predicted - actual
# SSE = sum(error^2)
# total_distance = sqrt(SSE)