clc;
clear variables;

% Initialize array
S = zeros(10000, 1);
sample_vector = zeros(10, 1);

% Randomly generate 10k numbers by the uniform distribution U[0, 1]
for itr = 1:10000
    S(itr) = rand;
end

% Do experiments 20 times: randomly pick up 10 numbers
for itr1 = 1:20
    select_number = randi([1 10000],1,10);
    
    % Calculate avg of 10 randomly pick-up numbers
    temp = 0;
    for itr2 = 1:10
        temp = temp + S(select_number(itr2));
    end
    
    sample_vector(itr1) = temp/10;
end

sample_mean = mean(sample_vector);
sample_devi = std(sample_vector);
