import matplotlib.pyplot as plt

iters = []
loss = []

with open('loss.log', 'r') as f:
    train_loss = f.readlines()
    for line in train_loss:
        tokens = line.split()
        print(tokens)
        iter = tokens[5][0:-1]
        iters.append(int(iter))
        loss.append(float(tokens[8]))

print(iters)
print(loss)

plt.plot(iters, loss, label='train loss', linewidth=2, color='r', marker='o', markerfacecolor='r', markersize=5)
plt.xlabel('Iters')
plt.ylabel('Loss Value')
plt.legend()
plt.savefig('vqa_lstm_loss.png')
#plt.show()
