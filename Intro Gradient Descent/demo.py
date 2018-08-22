import numpy as np
import sys

def cal_error_given_points(a, b, points):
    error = 0
    for i in range(len(points)):
        y = points[i,1]
        x = points[i,0]
        error += 0.5 * (y - (a * x + b)) ** 2
    return error / len(points)

def step_gredient_descent(a_curr, b_curr, points, learning_rate):
    a_gredient = a_curr
    b_gredient = b_curr
    N = len(points)
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        a_gredient += -(1/N) * (y - (a_curr * x + b_curr)) * (x)
        b_gredient += -(1/N) * (y - (a_curr * x + b_curr))
    a_next = a_curr - learning_rate * a_gredient
    b_next = b_curr - learning_rate * b_gredient
    return a_next, b_next


def greident_runner(init_a, init_b, points, num_iter, learning_rate):
    a_curr = init_a
    b_curr = init_b
    for i in range(len(points)):
        a_curr, b_curr = step_gredient_descent(a_curr, b_curr, points, learning_rate)
    return a_curr, b_curr

def main():
    filename = sys.argv[1]
    points = np.genfromtxt(filename, delimiter=",")
    learning_rate = 0.0001
    num_iter = 1000
    a = 0
    b = 0
    print("Starting gredient descent on dataset {0}, with a = {1}, b = {2}, total squared error = {3}" .format(filename, a, b, cal_error_given_points(a, b, points)))
    a, b = greident_runner(a, b, points, num_iter, learning_rate)
    print("Running")
    print("Starting gredient descent on dataset {0}, with a = {1}, b = {2}, total squared error = {3}".format(filename, a,b,cal_error_given_points(a, b, points)))

if __name__ == "__main__":
    main()



