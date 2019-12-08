import csv
import matplotlib.pyplot as plt
import numpy as np
import random

'''
Students:   Carlos Henrique Ponciano da Silva
            Jardel Angelo dos Santos
            Vinicius Luis da Silva
            
k) Which method is the most accurate?
Squared quadratic error (N = 3): 0.2477
To this dataset it seems to be the one the polynomial up to 3, because it's the regression measured with an unknown dataset
'''

def read_input() -> tuple:
    '''
    Reads the input off of a file with the path == 'Dados/data_preg.csv'
    and return a tuple with the x and y vectors in this order
    '''
    _x, _y = [], []

    with open('Dados/data_preg.csv') as data:
        _csv_reader = csv.reader(data)
    
        for row in _csv_reader:
            _x.append(float(row[0]))
            _y.append(float(row[1]))
    
    return (_x, _y)

def regression(x_vector : list, y_vector : list, bs : list) -> float:
    '''
    Given a x and y vector's alongside a beta vector (in reverse order) it calculates the value of the regression and returns de float result
    '''
    print(f'BetaN{len(bs)} = {bs}')
    _b0 = bs[-1]

    _counter = 1
    _buffer_result = []
    for i in range(len(bs) - 2, -1, -1):
        _buffer_result.append(np.dot(bs[i], np.power(x_vector, _counter)))
        _counter += 1

    _result = np.zeros(len(x_vector))

    for val in _buffer_result:
        _result = _result + val

    return _result + _b0

def eqm(residue : list, y_size : int) -> float:
    '''
    Calculates the squared quadratic error given a residue list and the regression vector size
    '''
    return sum(residue) / y_size

def residue(y_real : list, y_estimated : list) -> list:
    '''
    Calculates the residue vector given a vector of the real values and a vector of the predictions
    '''
    return [(_y_real - _y_estimated) ** 2 for (_y_real, _y_estimated) in zip(y_real, y_estimated)]

def split_data(x : list, y : list) -> tuple:
    '''
    Splits the x and y vectors into training and tests sets
    The split is done to a proportion of 90/10
    The split is random
    '''
    ten_percent_mark = int((len(y) / 100) * 10)
    idx_taining = sorted(random.sample(range(0, ten_percent_mark), ten_percent_mark))

    x_test = [x[idx] for idx in idx_taining]
    y_test = [y[idx] for idx in idx_taining]
    x_train = [x[idx] for idx in range(len(y)) if idx not in idx_taining] 
    y_train = [y[idx] for idx in range(len(y)) if idx not in idx_taining] 

    return (x_test, y_test, x_train, y_train)

def format_decimal(number : float) -> float:
    '''
    Formats a decimal number
    '''
    return '{0:.4f}'.format(number)

def plot(x : list, y : list, r1 : list, r2 : list, r3 : list, r8 : list, idx : int):
    '''
    Plots the data with 4 regression lines
    '''
    fig = plt.figure(idx)
    # plt.get_current_fig_manager().window.setGeometry(200 * idx, 250, 600, 600)
    plt.suptitle(f'Regressão Polinomial')
    plt.scatter(x, y, c='blue')
    plt.plot(x, r1, color='red')
    plt.plot(x, r2, color='green')
    plt.plot(x, r3, color='black')
    plt.plot(x, r8, color='yellow')

if __name__ == "__main__":
    x, y = read_input()

    x = np.array(x)
    y = np.array(y)

    # Calculates the betas for the ploting of the data
    bs1 = np.polyfit(x, y, 1)
    bs2 = np.polyfit(x, y, 2)
    bs3 = np.polyfit(x, y, 3)
    bs4 = np.polyfit(x, y, 4)
    
    # Calculates the regression for the ploting of the data
    r1 = np.array(regression(x, y, bs1))
    r2 = np.array(regression(x, y, bs2))
    r3 = np.array(regression(x, y, bs3))
    r8 = np.array(regression(x, y, bs4))

    # Plots the data
    plot(x, y, r1, r2, r3, r8, 1)

    # Calculates the squared quadratic error of the real data against the predictions 
    eqmr1 = eqm(residue(y, r1), len(y))
    eqmr2 = eqm(residue(y, r2), len(y))
    eqmr3 = eqm(residue(y, r3), len(y))
    eqmr8 = eqm(residue(y, r8), len(y))

    print(f'Squared Quadratic Error (N = 1): {format_decimal(eqmr1)}')
    print(f'Squared Quadratic Error (N = 2): {format_decimal(eqmr2)}')
    print(f'Squared Quadratic Error (N = 3): {format_decimal(eqmr3)}')
    print(f'Squared Quadratic Error (N = 8): {format_decimal(eqmr8)}')

    # Creates random training and test sets
    x_test_data, y_test_data, x_training_data, y_training_data = split_data(x, y)

    # Calculates de betas for the training data sets
    bs1 = np.polyfit(x_training_data, y_training_data, 1)
    bs2 = np.polyfit(x_training_data, y_training_data, 2)
    bs3 = np.polyfit(x_training_data, y_training_data, 3)
    bs4 = np.polyfit(x_training_data, y_training_data, 4)

    # Calculates de regressions for the training data sets
    r1 = np.array(regression(x_training_data, y_training_data, bs1))
    r2 = np.array(regression(x_training_data, y_training_data, bs2))
    r3 = np.array(regression(x_training_data, y_training_data, bs3))
    r8 = np.array(regression(x_training_data, y_training_data, bs4))

    # Plots the training dataset alongside with the regression lines
    plot(x_training_data, y_training_data, r1, r2, r3, r8, 5)

    # Calculates the regression lines of the test set using the betas calculated for the training set
    r1 = np.array(regression(x_test_data, y_test_data, bs1))
    r2 = np.array(regression(x_test_data, y_test_data, bs2))
    r3 = np.array(regression(x_test_data, y_test_data, bs3))
    r8 = np.array(regression(x_test_data, y_test_data, bs4))

    # Calculates the squared quadratic error of the test set 
    eqmr1 = eqm(residue(y_test_data, r1), len(y_test_data))
    eqmr2 = eqm(residue(y_test_data, r2), len(y_test_data))
    eqmr3 = eqm(residue(y_test_data, r3), len(y_test_data))
    eqmr8 = eqm(residue(y_test_data, r8), len(y_test_data))

    print(f'Squared Quadratic Error (N = 1): {format_decimal(eqmr1)}')
    print(f'Squared Quadratic Error (N = 2): {format_decimal(eqmr2)}')
    print(f'Squared Quadratic Error (N = 3): {format_decimal(eqmr3)}')
    print(f'Squared Quadratic Error (N = 8): {format_decimal(eqmr8)}')

    plt.show()
