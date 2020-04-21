clear variables;

x_0 = [1000; 1];
Q = [1 0; 0 1000];
p = [0; 0];
x_list = [x_0'];

f = @(x) 0.5*(x')*Q*x + (p')*x;
gradient_f = @(x) Q*x + p;
Hessian_f = @(x) Q;

% Newton's method
itr = 0;
while true
    if gradient_f(x_0) < eps
        break;
    else
        d_i = Hessian_f(x_0)\(-gradient_f(x_0));
        x_0 = x_0 + d_i;
        x_list = [x_list; x_0'];
    end
    itr = itr + 1;
end

final_x = x_0;
final_f_value = f(final_x);