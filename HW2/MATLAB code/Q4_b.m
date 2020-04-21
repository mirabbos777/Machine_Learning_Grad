clear variables;

% Training set from 4(a): positive := 1, negative := -1
A = zeros(13, 3);
A(1, :) =  [ 0    0    1];
A(2, :) =  [ 0.5  0    1];
A(3, :) =  [ 0    0.5  1];
A(4, :) =  [-0.5  0    1];
A(5, :) =  [ 0   -0.5  1];
A(6, :) =  [ 0.5  0.5 -1];
A(7, :) =  [ 0.5 -0.5 -1];
A(8, :) =  [-0.5  0.5 -1];
A(9, :) =  [-0.5 -0.5 -1];
A(10, :) = [ 1    0   -1];
A(11, :) = [ 0    1   -1];
A(12, :) = [-1    0   -1];
A(13, :) = [ 0   -1   -1];

% Perceptron algorithm (dual form)
% Initial parameters
alpha = zeros(size(A, 1), 1);
b = 0;
L = bigL(A(:, 1:2));
iter = 0;
mistake_flag = false;

% Update model
while true
    for i = 1:size(A, 1)
        % A(i, 1:2) := x^i and A(i, 3) := y_i
        % A(j, 1:2) := x^j and A(j, 3) := y_j
        sum = 0;
        for j = 1:size(A, 1)
            sum = sum + alpha(j, 1)*A(j, 3)*innerProduct(A(i, 1:2), A(j, 1:2));
        end
        if A(i, 3)*(sum + b) <= 0
            alpha(i, 1) = alpha(i, 1) + 1;
            b = b + A(i, 3)*L*L;
            mistake_flag = true;
            iter = iter + 1;
        end
    end
    if mistake_flag == false
        break;
    end
    mistake_flag = false;
end

% Randomly generate test set: x part \in [-1.5, 1.5], y part \in {-1, 1}
T = zeros(10000, 3);
a = -1.5;
b = 1.5;
for i = 1:size(T, 1)
    T(i, 1:2) = a + (b-a).*rand(1, 2);
    if randi([0 1]) == 0
        T(i, 3) = -1;
    else
        T(i, 3) = 1;
    end
end

% Classification result with test set
% for each 'result_i': (i=1~10000)
% component 1&2 := x part of test set
% component 3   := predicted value
% component 4   := true value
% collect correctly predicted results into 'correct_predict'
result = zeros(10000, 2);
correct_predict = [];
for i = 1:size(result, 1)
    result(i, 1:2) = T(i, 1:2);
    result(i, 3)   = perceptronDual(alpha, b, A, T(i, 1:2));
    result(i, 4)   = T(i, 3);
    if result(i, 3)>0 && result(i, 4)>0
        correct_predict = [correct_predict; result(i, :)];
    end
end

% plot figure
scatter(correct_predict(:, 1), correct_predict(:, 2), 'Marker', '.', 'DisplayName', 'h(x)>0 and +');
axis([-1.5 1.5 -1.5 1.5]);
xlabel('X Part Compoenet 1');
ylabel('X Part Compoenet 2');
hold on
scatter(A(1:5, 1), A(1:5, 2), 'MarkerFaceColor','r', 'MarkerEdgeColor', 'r', 'Marker', 'o', 'DisplayName', 'Postive Training Data');
scatter(A(6:13, 1), A(6:13, 2), 'MarkerFaceColor','b', 'MarkerEdgeColor', 'b', 'Marker', 'x', 'Linewidth', 2, 'SizeData' , 75, 'DisplayName', 'Negative Training Data');
lgd = legend;
set(gcf,'Position',[100 100 700 700]);
hold off
print('Q4_b_Figure', '-dpng', '-r800')
