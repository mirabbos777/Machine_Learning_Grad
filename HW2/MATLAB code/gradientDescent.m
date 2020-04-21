function [final_f_value, iter, final_x, x_list] = gradientDescent(Q, p, x0, tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Objective: min{ (1/2)*(x')*(Q)*(x) + (p')*(x) }
% Solve unconstrained minimization via gradient descent 
% with exact line search.
%
% Input argument: Must be a quadratic function!
% Q              coefficient matrix of degree-2 term
% p              coefficient matrix of degree-1 term
% x0             initial value to vector x
% tol            tolerance level to stop iteratively updating
%
% Output argument:
% final_f_value  function value at final step 
% iter           number of iterations 
% final_x        value of vector x at final step
% x_list         list all of vector x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize parameters
delta_x = 1;
iter = 0;
x_list = [x0'];

% Iteratively updating x
while delta_x > tol
    gradient_f = Q*x0 + p;
    temp1 = (gradient_f')*gradient_f;
    if temp1 < eps
        delta_x = tol;
    else
        stepsize = temp1/(gradient_f' * Q * gradient_f);
        x1 = x0 - stepsize*gradient_f;
        delta_x = norm(x1 - x0);
        x0 = x1;
        x_list = [x_list; x0'];
    end
    iter = iter + 1;
end

final_x = x0;
final_f_value = 0.5*(x0')*Q*x0 + (p')*x0;