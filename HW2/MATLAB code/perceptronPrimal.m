function [value] = perceptronPrimal(w, b, x)
value = dot(w, x) + b;