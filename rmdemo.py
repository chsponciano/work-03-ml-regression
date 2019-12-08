import matplotlib.pyplot as plt
import csv
import numpy as np
from statistics import mean
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D

'''
Students:   Carlos Henrique Ponciano da Silva
            Jardel Angelo dos Santos
            Vinicius Luis da Silva
'''
def read_input() -> tuple:
    '''
    Reads the input of data
    '''
    _sizes, _bedrooms, _prices = [], [], []

    with open('Dados/data.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            _sizes.append(int(row[0]))
            _bedrooms.append(int(row[1]))
            _prices.append(float(row[2]))

    return (_sizes, _bedrooms, _prices)

def _get_average(x : list, y : list) -> tuple:
    '''
    Returns a tuple with the avarege of both lists provided
    '''
    return mean(x), mean(y)

def build_data_structure(sizes : list, bedrooms : list, prices : list) -> tuple:
    '''
    Builds a data structure consisting of a tuple of two arrays
    the first one contains the values for the independent variables
    and se second one contains the values for the dependent variables
    '''
    _independent, _dependent = [], []
    
    for size, bedroom_count, price in zip(sizes, bedrooms, prices):
        _independent.append(np.array([1, size, bedroom_count]))
        _dependent.append(price)

    return (np.array(_independent), np.array(_dependent))

def correlation(x_vector : list, y_vector : list) -> float:
    '''
    Calculates the correlations of data betwen the x vector and the y vector
    '''
    _avarrege_x, _avarrege_y = _get_average(x_vector, y_vector)
    
    dividend = sum([(x - _avarrege_x) * (y - _avarrege_y) for x, y in zip(x_vector, y_vector)])
    divider = sqrt(sum([(x - _avarrege_x) ** 2 for x in x_vector]) * sum([(y - _avarrege_y) ** 2 for y in y_vector]))
    
    return dividend / divider

def regression(independent : list, beta : list) -> list:
    '''
    Calculates the regression given independent a matrix of independent variables and a beta valuw
    '''
    return np.dot(independent, beta)

def calculete_beta(independent : list, dependent : list) -> list:
    '''
    Calculates a beta value given a matrix of independent and a vector of dependent values
    '''
    _independent_transpose = np.transpose(independent)
    _independent_inverted = np.linalg.inv(np.dot(_independent_transpose, independent))
    return np.dot(_independent_inverted, np.dot(_independent_transpose, dependent))

def format_decimal(number : float) -> float:
    '''
    Formats a decimal number
    '''
    return '{0:.4f}'.format(number)

def plot(x : list, y : list, z : list, r : list, data : tuple):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Sizes', fontsize=10)
    ax.set_ylabel('Bedrooms', fontsize=10)
    ax.set_zlabel('Prices', fontsize=10)

    ax.plot3D(x, y, r, color='green')
    ax.scatter3D(x, y, z, c='b')
    
    plt.suptitle('Multiple linear regression')
    plt.title(f'Correlation sizes/prices: {format_decimal(data[0])}', loc='left', fontsize=10)
    plt.title(f'Correlation bedrooms/prices: {format_decimal(data[1])}', loc='right', fontsize=10)
    plt.title(f'The price of a house with size 1650 e 3 bedrooms: {format_decimal(data[2])}', y=-0.15, x=0.30, fontsize=10)

    plt.show()

if __name__ == "__main__":
    _sizes, _bedrooms, _prices = read_input()

    # Correlation sizes/prices
    _csp = correlation(_sizes, _prices)

    # Correlation bedrooms/prices
    _cbp = correlation(_bedrooms, _prices)

    # Multiple Regression
    _independent, _dependent = build_data_structure(_sizes, _bedrooms, _prices)
    _beta = calculete_beta(_independent, _dependent)
    _y = regression(_independent, _beta)

    # Calculating de price of a house with size 1650 e 3 bedrooms
    # Result should be 293081
    _independent = np.array([1, 1650, 3])
    _question_g = regression(_independent, _beta)
    
    plot(_sizes, _bedrooms, _prices, _y, (_csp, _cbp, _question_g))