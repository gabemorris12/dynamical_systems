#These ODE solvers are based on code in chapter 7 of Numerical methods in engineering with python (see digital copy in bookmarks and in files)
import numpy
def Euler(F,x,y,xStop,h):
    X = []
    Y = []
    X.append(x)  #vecor of points for independent variable, currently only with initial point
    Y.append(y)  #vector of points for dependent variables; currently the vector is a row with each column representing one of the initial values.
    while x < xStop:
        h = min(h,xStop - x)
        y = y + h*F(x,y)
        x = x + h
        X.append(x)
        Y.append(y)
    return numpy.array(X),numpy.array(Y)

#Define function to perform integration for 4th order runge kutta method.
def RK4(F,x,y,xStop,h):
    def run_kut4(F,x,y,h):   #Defining function for finding the incremental change at a given step.
        K0 = h*F(x,y)
        K1 = h*F(x+h/2.0, y+K0/2.0)
        K2 = h*F(x+h/2.0, y+K1/2.0)
        K3 = h*F(x+h, y+K2)
        return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:       #while loop applies incremental change to each step.
        h = min(h,xStop - x)
        y = y+run_kut4(F,x,y,h)
        x = x+h
        X.append(x)
        Y.append(y)
    return numpy.array(X), numpy.array(Y)