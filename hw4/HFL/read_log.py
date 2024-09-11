import argparse
import os
from matplotlib.pylab import plt
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    parser.add_argument("--algorithm", type=str, default="pFedMe")
    args = parser.parse_args()

    # print("alpha = 0.1: ", np.random.dirichlet(np.repeat(0.1, 10)))
    # print("alpha = 50: ", np.random.dirichlet(np.repeat(50, 10)))
    path = os.path.join(args.result_path, args.algorithm, "models", args.dataset, "training.log")
    acc= []
    loss = []
    path = "training.log"
    with open(path, 'r') as f:
        for line in f:
            # print(line)
            words = line.split()
            # print(words)
            if len(words) > 2:
                if words[2] == 'Average':
                    acc.append(float(words[-4][:-1]))
                    if float(words[-1][:-1]) > 10:
                        loss.append(loss[-1])
                    else:
                        loss.append(float(words[-1][:-1]))
    
    rounds = range(len(acc))
    plt.plot(rounds, acc, label='Accuracy Curve')
    plt.title('num users = 10')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()

    plt.plot(rounds, loss, label='Loss Curve')
    plt.title('num users = 10')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

