clc;
clear variables;

A = [1 2; 2 -1; 1 1; 4 -1; 1 -1; 1 -2];
b = [2; -2; 1; 1; -2; -1];

func_1 = @(x) norm(A*x-b, 1);
func_2 = @(x) norm(A*x-b, 2);
func_inf = @(x) norm(A*x-b, Inf);

config = optimoptions(@fminunc, 'Display', 'off', 'Algorithm', 'quasi-newton', 'HessUpdate', 'steepdesc');

ini_x = [-3; -3];
[x_1, func_1_val] = fminunc(func_1, ini_x, config);
[x_2, func_2_val] = fminunc(func_2, ini_x, config);
[x_inf, func_inf_val] = fminunc(func_inf, ini_x, config);
ans_1 = [x_1; func_1_val];
ans_2 = [x_2; func_2_val];
ans_inf = [x_inf; func_inf_val];

for i = -2.9:0.1:3
    for j = -2.9:0.1:3
        ini_x = [i; j];
        [x_1, func_1_val] = fminunc(func_1, ini_x, config);
        [x_inf, func_inf_val] = fminunc(func_inf, ini_x, config);

        if ans_1(3, 1) > func_1_val
            ans_1 = [x_1; func_1_val];
        end
        
        if ans_inf(3, 1) > func_inf_val
            ans_inf = [x_inf; func_inf_val];
        end
    end
end

% Draw
eq1 = @(x) (2-x)/2;
eq2 = @(x) 2*x+2;
eq3 = @(x) 1-x;
eq4 = @(x) 4*x-1;
eq5 = @(x) x+2;
eq6 = @(x) (x+1)/2;

fplot(eq1);

hold on;

xlim([-3, 3]);
ylim([-3, 3]);
grid on;
xL = xlim;
yL = ylim;
line([0 0], yL, 'Color', 'k', 'LineWidth', 0.2);  %x-axis
line(xL, [0 0], 'Color', 'k', 'LineWidth', 0.2);  %y-axis

fplot(eq2);
fplot(eq3);
fplot(eq4);
fplot(eq5);
fplot(eq6);

plot(ans_1(1, 1), ans_1(2, 1), '.', 'MarkerSize', 10);
plot(ans_2(1, 1), ans_2(2, 1), '.', 'MarkerSize', 10);
plot(ans_inf(1, 1), ans_inf(2, 1), '.', 'MarkerSize', 10);

hold off;
print('Figure_Q6', '-dpng', '-r400');