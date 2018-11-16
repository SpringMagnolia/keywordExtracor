import os 
import codecs


def merge():
    temp_list = [i for i in os.listdir(r"./") if i.endswith(".txt")]
    all_list = []
    for temp in  temp_list:
        all_list.extend(codecs.open(temp).readlines())
    all_list = [i.strip() for i in set(all_list) if len(i)<=5]
    with open("all.txt","a") as f:
        for t in all_list:
            f.write(t+"\n")


if __name__ == "__main__":
    merge()
