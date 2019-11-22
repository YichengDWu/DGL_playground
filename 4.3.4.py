# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:30:33 2019

@author: lenovo
"""

#
import numpy as np

def formula_5(f, x, h):
    return 1./(2*h) * (f(x + h) - f(x - h))

def formula_19(f, x, h):
    return formula_5(f, x, h)- 1./ (12*h) * (f(x+2*h) - 2 * (f(x+h) - f(x - h)) - f(x-2 * h))

x = 0.
f = np.sin
print('\t\t\t     h                  formula_5\t\t      formula_19')
print('------------------------------------------------------------------------------------------')
for n in range(13):
    h = 4 ** (-n) 
    error_1 = formula_5(f, x, h)-1.
    error_2 = formula_19(f, x, h)-1.
    print("{:30}{:30}{:30}".format(h , error_1, error_2))