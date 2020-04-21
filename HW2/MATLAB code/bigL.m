function [value] = bigL(x)
max_value = 0;

for i = 1:size(x, 1)
    max_value = max(max_value, norm(x(i, :))^2);
end

value = max_value;