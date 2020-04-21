clc;
clear variables;

Q = [1 0; 0 1000];
p = [0; 0];
x_0 = [1000; 1];

[f_value, iter, x, x_list] = gradientDescent(Q, p, x_0, (10^-8));