from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import os.path
from os import walk
import pickle


path = '..\\..\\Results\\'
def get_format_name(attempt, version, file_format, date = datetime.now().date()): 
    return f'result_{version}_{attempt}_{date}{file_format}'

#file formats
model_file_format = '.model'
info_file_format = '.info'
jpg_file_format = '.jpg'


def save_model(result : GridSearchCV):
    attempt, version = 0, 0
    with open(os.path.join(path, get_format_name(attempt, version, model_file_format)), 'xb') as model:
        model.write(pickle.dumps(result))


def load_model(attempt, version):
    # if not os.path.isfile(os.path.join(path, file_name)):
    #     raise Exception('This file doesn\'t exist')
    filenames = next(walk(path))[2]
    for filename in filenames:
        if get_format_name(attempt, version, '', '') in filename:
            date = filename[len(get_format_name(attempt, version, '', '')):len(get_format_name(attempt, version, '', '')) + 10]
            print(date)
            break
    else:
        raise Exception('This file doesn\'t exist')

    with open(os.path.join(path, get_format_name(attempt, version, model_file_format, date)), 'rb') as model:
        result = pickle.loads(model.read())
    
    return result