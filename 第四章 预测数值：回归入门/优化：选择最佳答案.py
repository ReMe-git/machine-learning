from mlwpy import *

# 为选取m和b的最佳值或者权重的最佳值
tgt = np.array([3, 5, 8, 10, 12, 15])

# 随机猜测
#num_gussess = 10
#results = []

#or g in range(num_gussess):
#    guess = np.random.uniform(low=tgt.min(), high=tgt.max())
#    total_dist = np.sum((tgt - guess)**2)
#    results.append((total_dist, guess))
#best_guess = sorted(results)[0][1]
#print("随机猜测：", best_guess)

# 随机调整
#num_steps = 100
#step_size = 0.05

#best_guess = np.random.uniform(low=tgt.min(), high=tgt.max())
#best_dist = np.sum((tgt - best_guess)**2)

#for s in range(num_steps):
#    new_guess = best_guess + (np.random.choice([+1, -1]) * step_size)
#    new_dist = np.sum((tgt - best_guess)**2)
#    if new_dist < best_dist:
#       best_guess, best_dist = new_guess, new_dist

#print("随机调整：", best_guess)

# 智能调整
num_steps = 1000
step_size = 0.02

best_guess = np.random.uniform(low=tgt.min(), high=tgt.max())
best_dist = np.sum((tgt - best_guess)**2)

for s in range(num_steps):
    guesses = best_guess + (np.array([-1, 1]) * step_size)
    dists = np.sum((tgt[:, np.newaxis] - guesses)**2, axis=0)

    better_idx = np.argmin(dists)

    if dists[better_idx] > best_dist:
        break

    best_guess = guesses[better_idx]
    best_dist = dists[better_idx]

print("智能调整：", best_guess)