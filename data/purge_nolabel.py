import os
import csv
if __name__ == '__main__':
    fr = open('val_mani.csv', 'r')
    fw = open('purged_val.csv', 'w')
    for item in fr.readlines():
        itemstr = item.strip().split(',')
        if os.path.exists(itemstr[0]) and os.path.exists(itemstr[1]):
            fw.write(item)
        else:
            print(item)

