function [x1, x2, x3, x4] = nonlinearMap(x)
x1 = -x(1, 1)*x(1, 2);
x2 = x(1, 1)*x(1, 1);
x3 = x(1, 1)*x(1, 2);
x4 = x(1, 2)*x(1, 2);