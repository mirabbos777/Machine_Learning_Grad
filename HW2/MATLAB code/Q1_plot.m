clear variables;

X = [0; 0; 0; 0; 0; 0.1; 0.1; 0.1; 0.2; 0.2; 0.3; 0.3; 0.4; 0.4; 0.5; 0.6; 0.7; 0.8; 0.8; 0.9; 1];
Y = [0; 0.1; 0.2; 0.3; 0.4; 0.4; 0.5; 0.6; 0.6; 0.7; 0.7; 0.8; 0.8; 0.9; 0.9; 0.9; 0.9;0.9; 1; 1; 1];

hold on;

ROC_curve = plot(X, Y);
ROC_curve.Color = '#0072BD';
ROC_curve.LineWidth = 2;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
axis([-0.001 1.001 -0.001 1.001]);
ROC_curve.Marker = '.';
ROC_curve.MarkerSize = 20;

AUC = area(X, Y);
AUC.FaceAlpha = 0.2;
AUC.FaceColor = ROC_curve.Color;
AUC.FaceAlpha = 0.125;

AUC_value = trapz(X, Y);
AUC_value = num2str(AUC_value, '%.4f');
txt = strcat('AUC = ',AUC_value);
t = text(0.62, 0.14, txt);
t.FontSize = 15;
t.Color = 'r';
print('Figure_Q1', '-dpng', '-r600')

hold off;