clc;
clear variables;

% Initialize array
S = zeros(10000, 1);
sample_vector = zeros(10, 1);

% Randomly generate 10k numbers by the uniform distribution U[0, 1]
for itr = 1:10000
    S(itr) = rand;
end

% Do experiments 50 times: randomly pick up 10 numbers
for itr1 = 1:50
    select_number = randi([1 10000],1,1000);
    
    % Calculate avg of 10 randomly pick-up numbers
    temp = 0;
    for itr2 = 1:1000
        temp = temp + S(select_number(itr2));
    end
    
    sample_vector(itr1) = temp/1000;
end

sample_mean = mean(sample_vector);
sample_devi = std(sample_vector);
