clc;
clear variables;

% Initialize input arrays
X = zeros(9, 2); % X part of data is a 2D vector

% Store input data into arrays
X(1, 1:2) = [0 0];
X(2, 1:2) = [1 1];
X(3, 1:2) = [-1 1];
X(4, 1:2) = [1 -1];
X(5, 1:2) = [-1 -1];
X(6, 1:2) = [1 0];
X(7, 1:2) = [-1 0];
X(8, 1:2) = [0 1];
X(9, 1:2) = [0 -1];

Y = { 'positive'; 'positive'; 'positive'; 'positive'; 'positive'; 
      'negative'; 'negative'; 'negative'; 'negative'};

% Apply 1-nearest neighbor algorithm
Model = fitcknn(X,Y,'NumNeighbors',1);

% Plot's configuration
x_range = min(X(:,1)):.005:max(X(:,1));
y_range = min(X(:,2)):.005:max(X(:,2));
[enum_x, enum_y] = meshgrid(x_range,y_range);
XGrid = [enum_x(:) enum_y(:)];

% Plot the results
predict_result = predict(Model, XGrid);
gscatter(enum_x(:), enum_y(:), predict_result,'rgb', '.');
