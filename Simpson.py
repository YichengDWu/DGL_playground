# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:17:28 2019

@author: lenovo
"""
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
plt.style.use('dark_background')


def f1(x):
    return 4./(x**2+1)

def f2(x):
    return 8*(np.sqrt(1-x**2)-x)

def Simpson(f,a,b,eps, level, level_max):    
    level += 1
    h = b-a
    c = (a+b)/2
    one_simpson = h * (f(a)+4 * f(c) + f(b))/6
    d = (a+c)/2
    e = (c+b)/2
    two_simpson = h * (f(a) + 4* f(d)+2*f(c)+4*f(e)+f(b))/12
    if level>= level_max:
        print("maximum level reached!")
        print("Parition: [{:6},{:6},{:6}]".format(a,c,b))
        paritions.append(c)
        return two_simpson
    else:
        if abs(two_simpson - one_simpson) < 15*eps:
            paritions.append(c)
            result = two_simpson+(two_simpson-one_simpson)/15
        else:
            left_simpson = Simpson(f,a,c,eps/2, level, level_max)
            right_simpson = Simpson(f,c,b,eps/2,level, level_max)
            result = left_simpson + right_simpson
            
    return result


def plot(fig, ax, paritions,f,a,b, ylim0, ylim1, fname = ''):
    x, y = [],[]
    sc = ax.scatter(x,y,c='r')
    plt.xlim(a-0.04,b + .04)
    plt.ylim(ylim0-0.04,ylim1+.04)

    x_line = np.linspace(0, 1., 200)
    y_line = f(x_line)

    ax.plot(x_line, y_line)
    ax.scatter([a,b], [f(0),f(1)], c = 'r')

    def animate(i):
        x.append(paritions[i])
        y.append(f(paritions[i]))
        
        ax.set_title("{}\nStep = {}, c = {:06}".format(fname,i,paritions[i]),
                              color='g', size = 16)
        sc.set_offsets(np.c_[x,y])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ani = matplotlib.animation.FuncAnimation(fig, animate, 
                frames=np.arange(len(paritions)), interval=700, 
                                         repeat=False)  
    return ani

fig1, ax1 = plt.subplots()

print("Function 1")
paritions = []
result = Simpson(f1,0,1,1e-9, 0, 4)
ani = plot(fig1, ax1,paritions,f1,0,1,0, 4, 'f(x) = 4/(1+x^2)')
plt.show()

print("I = ",result)

fig2, ax2 = plt.subplots()

print("Function 2")
paritions = []
result = Simpson(f2,0,1./np.sqrt(2),1e-9, 0, 4)
ani2 = plot(fig2, ax2,paritions,f2,0,1./np.sqrt(2),0, 8,'f(x) = 8(sqrt(1-x^2)-1)')
plt.show()
print("I = ",result)
