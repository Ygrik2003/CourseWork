from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import os.path
from os import walk
import pickle
import re

path = '..\\..\\Results\\'
def get_format_name(version, attempt, file_format, date = datetime.now().date()): 
    return f'result_{version}_{attempt}_{date}{file_format}'

#file formats
model_file_format = '.model'
info_file_format = '.info'
jpg_file_format = '.jpg'


def save_model(result : GridSearchCV, version=None):
    last_version = -1
    last_attempt = -1

    filenames = next(walk(path))[2]
    for filename in filenames:
        match = re.search('result_(\d+)_(\d+)', filename)
        if version == None:
            if last_version < int(match[1]):
                last_version = int(match[1])

        elif version == int(match[1]):
                last_attempt = max(last_attempt, int(match[2]))
                
    version = version if version != None else last_version + 1
    attempt = last_attempt + 1
    with open(os.path.join(path, get_format_name(version, attempt, model_file_format)), 'xb') as model:
        model.write(pickle.dumps(result))


def load_model(version, attempt):
    filenames = next(walk(path))[2]
    for filename in filenames:
        match = re.search('result_(\d+)_(\d+)_(\d{4})-(\d{2})-(\d{2})', filename)
        if int(match[1]) == version and int(match[2]) == attempt:
            date = datetime(int(match[3]), int(match[4]), int(match[5]))
            break
    else:
        raise Exception('This file doesn\'t exist')

    with open(os.path.join(path, get_format_name(version, attempt, model_file_format, date.date())), 'rb') as model:
        result = pickle.loads(model.read())
    
    return result