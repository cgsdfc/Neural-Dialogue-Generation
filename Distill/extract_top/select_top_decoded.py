import argparse
import collections

MAX_DECODED_NUM = 1000000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    A = open(args.input_file, "r")
    lines = A.readlines()
    print("num of lines " + str(len(lines)))
    Dict = collections.defaultdict(int)
    count = 0

    for line in lines:
        t1 = line.find(" ")
        t2 = line.find(" ", t1 + 1)
        t3 = line.find(" ", t2 + 1)
        line = line[t3 + 1:].strip()

        Dict[line] += 1
        count += 1
        if count == MAX_DECODED_NUM:
            break

    A = open(args.output_file, "w")

    sort = sorted(Dict.items(), key=lambda x: x[1], reverse=True)
    t = 0
    for item in sort:
        print(item[1], '|', item[0], sep='', file=A)
        t = t + 1
        if item[1] <= 50:
            break
