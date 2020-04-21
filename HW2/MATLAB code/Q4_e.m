clear variables;

% Training set
% From 4(a): positive := 1, negative := -1
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

% Test set
% Randomly generate: x part \in [-1.5, 1.5], y part \in {-1, 1}
T = zeros(10000, 3);
a = -1.5;
b = 1.5;
for i = 1:size(T, 1)
    T(i, 1:2) = a + (b-a).*rand(1, 2);
    if randi([0 1]) == 0
        T(i, 3) = -1;% negative instance
    else
        T(i, 3) = 1;% positive instance
    end
end

% Non-linearly map A & T into feature space with higher dimension
high_dim_A = zeros(13, 5);
high_dim_T = zeros(10000, 5);
for i = 1:size(A, 1)
    [high_dim_A(i, 1), high_dim_A(i, 2), high_dim_A(i, 3), high_dim_A(i, 4)] = nonlinearMap(A(i, 1:2));
    high_dim_A(i, 5) = A(i, 3);
end
for i = 1:size(T, 1)
    [high_dim_T(i, 1), high_dim_T(i, 2), high_dim_T(i, 3), high_dim_T(i, 4)] = nonlinearMap(T(i, 1:2));
    high_dim_T(i, 5) = T(i, 3);
end

% Perceptron algorithm (primal form)
% Initial parameters
eta = rand()+eps;
w = zeros(4, 1);
b = 0;
R = bigR(high_dim_A(:, 1:4));
iter = 0;
mistake_flag = false;
update_history = [];

% Update model
while true
    for i = 1:size(high_dim_A, 1)
        % high_dim_A(i, 1:4) := x^i and high_dim_A(i, 5) := y_i
        if high_dim_A(i, 5)*(dot(w', high_dim_A(i, 1:4)) + b) <= 0
            old_w = w;
            old_b = b;
            w = w + eta*high_dim_A(i, 5)*(high_dim_A(i, 1:4)');
            b = b + eta*high_dim_A(i, 5)*R;
            mistake_flag = true;
            iter = iter + 1;
            update_history = [update_history;iter old_w' old_b w' b];
        end
    end
    if mistake_flag == false
        break;
    end
    mistake_flag = false;
end

% Classifying test set
% for each 'result_i' (i=1~10000)
% component 1&2 := x part of test set
% component 3   := predicted value
% component 4   := true value
% collect correctly predicted results into 'correct_predict'
result = zeros(10000, 4);
correct_predict = [];
for i = 1:size(result, 1)
    result(i, 1:2) = T(i, 1:2);
    result(i, 3)   = perceptronPrimal(w, b, high_dim_T(i, 1:4));
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
print('Figure_Q4_e', '-dpng', '-r800')