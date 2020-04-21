function [value] = perceptronDual(alpha, b, trainSet, x)
value = 0;
for i = 1:size(trainSet, 1)
    value = value + alpha(i, 1)*trainSet(i, 3)*(trainSet(i, 1:2)*x');
end
value = value + b;