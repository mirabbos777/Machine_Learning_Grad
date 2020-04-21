clear variables;

% Training set
% From 4(c): positive := 1, negative := -1
B = zeros(8, 3);
B(1, :) =  [ 0.5  0    1];
B(2, :) =  [ 0  0.5    1];
B(3, :) =  [-0.5  0    1];
B(4, :) =  [ 0   -0.5  1];
B(5, :) =  [ 0.5  0.5 -1];
B(6, :) =  [ 0.5 -0.5 -1];
B(7, :) =  [-0.5  0.5 -1];
B(8, :) =  [-0.5 -0.5 -1];

% Test set
% Randomly generate: x part \in [-1.5, 1.5], y part \in {-1, 1}
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

% Perceptron algorithm (dual form)
% Initial parameters
alpha = zeros(size(B, 1), 1);
b = 0;
L = bigL(B(:, 1:2));
iter = 0;
mistake_flag = false;

% Update model
while true
    for i = 1:size(B, 1)
        % B(i, 1:2) := x^i and B(i, 3) := y_i
        % B(j, 1:2) := x^j and B(j, 3) := y_j
        sum = 0;
        for j = 1:size(B, 1)
            sum = sum + alpha(j, 1)*B(j, 3)*innerProduct(B(i, 1:2), B(j, 1:2));
        end
        if B(i, 3)*(sum + b) <= 0
            alpha(i, 1) = alpha(i, 1) + 1;
            b = b + B(i, 3)*L*L;
            mistake_flag = true;
            iter = iter + 1;
        end
    end
    if mistake_flag == false
        break;
    end
    mistake_flag = false;
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
    result(i, 3)   = perceptronDual(alpha, b, B, T(i, 1:2));
    result(i, 4)   = T(i, 3);
    if result(i, 3)>0 && result(i, 4)>0
        correct_predict = [correct_predict; result(i, :)];
    end
end

scatter(correct_predict(:, 1), correct_predict(:, 2), 'Marker', '.', 'DisplayName', 'Test Instances: h(\bfx\rm)>0 and +');
axis([-1.5 1.5 -1.5 1.5]);
xlabel('\bfx^{i}\rm Part Compoenet 1');
ylabel('\bfx^{i}\rm Part Compoenet 2');
hold on
scatter(B(1:4, 1), B(1:4, 2), 'MarkerFaceColor','r', 'MarkerEdgeColor', 'r', 'Marker', 'o', 'DisplayName', 'Postive Training Data');
scatter(B(5:8, 1), B(5:8, 2), 'MarkerFaceColor','b', 'MarkerEdgeColor', 'b', 'Marker', 'x', 'Linewidth', 2, 'SizeData' , 75, 'DisplayName', 'Negative Training Data');
lgd = legend;
set(gcf,'Position',[100 100 700 700]);
hold off
print('Figure_Q4_c', '-dpng', '-r600')
