clc;
clear variables;

% Initialize array
S_X = zeros(1000, 2);
S_Y = zeros(1000, 1);
error_vector = zeros(1000, 1);

% Randomly generate 1k data sets by U[-1, 1] and std. normal distribution
lower_bound = -1;
upper_bound = 1;

for itr = 1:1000
    temp_x1 = lower_bound + ( upper_bound - lower_bound ) * rand;          % x_1 part
    temp_x2 = lower_bound + ( upper_bound - lower_bound ) * rand;          % x_2 part
    S_X(itr, 1:2) = [temp_x1 temp_x2];
    S_Y(itr, 1) = (2 * temp_x1 * temp_x1) + (temp_x2 * temp_x2) - (2 * temp_x1 * temp_x2) + (2 * temp_x1) - temp_x2 + randn;
end

fit_surface = fit([S_X(:, 1), S_X(:, 2)], S_Y, 'poly22');
plot(fit_surface, [S_X(:, 1), S_X(:, 2)], S_Y);

% Calculate the error vector and MAE of the surface
error_vector = fit_surface(S_X(:, 1), S_X(:, 2)) - S_Y;                    % error = predicted value - true value
MAE = mae(error_vector);
