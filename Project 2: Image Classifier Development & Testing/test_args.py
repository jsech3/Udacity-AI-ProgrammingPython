import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()

args = get_input_args()
print("Data directory:", args.data_dir)
print("Use GPU:", args.gpu)
