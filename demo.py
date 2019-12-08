import matplotlib.pyplot as plt
from math import sqrt
from statistics import mean 
import numpy as np

'''
Students:   Carlos Henrique Ponciano da Silva
            Jardel Angelo dos Santos
            Vinicius Luis da Silva
            
3) Which of the datasets isn't appropriate for a liner regression?
The third one, because the independent variable doesn't seem to affect the dependent variable

'''
def _get_average(x : list, y : list) -> tuple:
    '''
    Returns a tuple with the avarege of both lists provided
    '''
    return mean(x), mean(y)

def correlation(x_vector : list, y_vector : list) -> float:
    '''
    Calculates the correlations of data betwen the x vector and the y vector
    '''
    _avarrege_x, _avarrege_y = _get_average(x_vector, y_vector)
    
    dividend = sum([(x - _avarrege_x) * (y - _avarrege_y) for x, y in zip(x_vector, y_vector)])
    divider = sqrt(sum([(x - _avarrege_x) ** 2 for x in x_vector]) * sum([(y - _avarrege_y) ** 2 for y in y_vector]))
    
    return dividend / divider

def regression(x_vector : list, y_vector : list) -> tuple:
    '''
    Returns the result for the regression and the values of b0 and b1
    '''
    _avarrege_x, _avarrege_y = _get_average(x_vector, y_vector)

    _b1 = calculate_beta_1(x_vector, y_vector, _avarrege_x, _avarrege_y)
    _b0 = calculate_beta_0(_avarrege_x, _avarrege_y, _b1)
   
    return ([_b0 + _b1 * _x for _x in x_vector], _b0, _b1)
    

def calculate_beta_1(x_vector : list, y_vector : list, avarrege_x : float, avarrege_y : float) -> float:
    '''
    Calculates the value for b1
    '''
    dividend = sum([(x - avarrege_x) * (y - avarrege_y) for x, y in zip(x_vector, y_vector)])
    divider = sum([(x - avarrege_x) ** 2 for x in x_vector])

    return dividend / divider

def calculate_beta_0(avarrege_x : float, avarrege_y : float, b1 : float) -> float:
    '''
    Calculates the value for b0
    '''
    return avarrege_y - (b1 * avarrege_x)

def format_decimal(number : float) -> float:
    '''
    Formats a decimal number
    '''
    return '{0:.4f}'.format(number)

def plot(x : list, y : list, r : list, idx : int, data : tuple):
    fig = plt.figure(idx+1)
    # plt.get_current_fig_manager().window.setGeometry(100 * (idx * 6) + 65, 250, 600, 600)
    plt.suptitle('Linear Regression')
    plt.title(f'Correlation: {format_decimal(data[0])} ', loc='left')
    plt.title(f'B0: {format_decimal(data[1])} ', loc='center')
    plt.title(f'B1: {format_decimal(data[2])} ', loc='right')
    plt.scatter(x, y)   
    plt.plot(x, r, color='green')

if __name__ == "__main__":
    _data_sets = [
        {
            'x' : [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            'y' : [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        },
        {
            'x' : [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            'y' : [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
        },
        {
            'x' : [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19],
            'y' : [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]
        }        
    ]
    
    for i in range(len(_data_sets)):
        _x, _y = np.array(_data_sets[i]['x']), np.array(_data_sets[i]['y'])
        _c = correlation(_x, _y)
        _r, _b0, _b1 = regression(_x, _y)
        plot(_x, _y, _r, i, (_c, _b0, _b1))

    plt.show()