import numpy as np
import matplotlib.pyplot as plt
import csv

colours = ['#99C000', '#FDCA00']
labels = ['Ours', 'TCN']
# files = ["/home/kluger/Downloads/run_32_b4_190313-110919_tensorboard-tag-train_max_err_loss_avg.csv",
#          "/home/kluger/Downloads/run_14_18_b4_190325-140449_tensorboard-tag-train_max_err_loss_avg.csv"]
files = ["/home/kluger/Downloads/run_32_b4_190313-110919_tensorboard-tag-val_max_err_loss_avg.csv",
         "/home/kluger/Downloads/run_14_18_b4_190325-140449_tensorboard-tag-val_max_err_loss_avg.csv"]

plt.figure(figsize=(5,3.8))

loss_curves = []

for fi, file in enumerate(files):
    print(file)
    colour = colours[fi]

    loss_curve = []

    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:

            step = int(row[1])
            loss = float(row[2])
            # print(step, loss)
            point = np.array([step, loss])
            loss_curve += [point]

    loss_curve = np.vstack(loss_curve)
    loss_curves += [loss_curve]

    print(loss_curve.shape)

    plt.plot(loss_curve[:,0]/1000, loss_curve[:,1], '-', lw=1.5, c=colour, label=labels[fi])

plt.ylim(0, 0.08)
plt.xlim(0, 40)
plt.grid(lw=0.2, color='0.25')
plt.legend(loc='upper right', fontsize='small')
plt.ylabel("loss")
plt.xlabel("iterations (1e3)")
plt.show()