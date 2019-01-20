# coding: utf-8

from __future__ import print_function
import json
import argparse

def json_file_read(path):
    """ 
        load data from json file 
    """
    with open(path) as data_file:
        data = json.load(data_file)
    return data

if __name__ == "__main__":
    # set command line argument 
    parser = argparse.ArgumentParser("ADD PATH")
    parser.add_argument("path",
        default = "",
        help = "path to json file")
    # load the data
    opt = parser.parse_args()
    data = json_file_read(opt.path)


