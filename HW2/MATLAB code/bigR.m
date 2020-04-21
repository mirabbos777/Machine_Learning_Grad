function [value] = bigR(x)
max_value = 0;

for i = 1:size(x, 1)
    max_value = max(max_value, norm(x(i, :)));
end

value = max_value;