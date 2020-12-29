import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--task_name', default='mab')
parser.add_argument('-f', '--file_name', default='feat', help='file name')
parser.add_argument('-d', '--d_model', default=128, help='dimension of the model')
parser.add_argument('-b', '--batch_size', default=128, help='batch_size')
parser.add_argument('-m', '--max_iter', default=200000, help='max iter')
parser.add_argument('-fc', '--dff', default=128)
parser.add_argument('-hh', '--n_head', default=4)
parser.add_argument('-l', '--learning_rate', default=0.001)
parser.add_argument('-t', '--max_sec', default=36000)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
