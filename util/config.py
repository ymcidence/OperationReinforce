import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--task_name', default='basic1')
parser.add_argument('-d', '--d_model', default=128, help='dimension of the model')
parser.add_argument('-b', '--batch_size', default=128, help='batch_size')
parser.add_argument('-nh', '--n_head', default=4)
parser.add_argument('-lr', '--learning_rate', default=0.001)
parser.add_argument('-mt', '--max_time', default=6)
parser.add_argument('-rr', '--rate', default=.9)
parser.add_argument('-nl', '--n_layer', default=4)
parser.add_argument('-tt', '--temp', default=1.9)
parser.add_argument('-st', '--sample_time', default=10)
parser.add_argument('-npl', '--exp_n_pl', default=1.7)
parser.add_argument('-ee', '--epoch', default=100)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
