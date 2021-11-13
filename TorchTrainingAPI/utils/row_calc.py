import os
import argparse
my_parser = argparse.ArgumentParser(description='List the content of a folder')
my_parser.add_argument('--path',
                            type=str)
args = my_parser.parse_args()
ROOT_PATH = args.path
COUNT_TYPE = ['py']
if __name__ == '__main__':
    lines = 0
    for filepath, dirnames, filenames in os.walk(ROOT_PATH):
        for filename in filenames:
            path = os.path.join(filepath, filename)
            type = filename.split(".")[-1]
            if(type in COUNT_TYPE):
                count = len(open(path,encoding='UTF-8').readlines())
                print(path,",lines:",count)
                lines += count
    print("total count :",lines)