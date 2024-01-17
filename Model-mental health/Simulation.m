% Load the dataset.
data = randi([1, 5], 10000, 90);

% Define the order of the questions.
problem_order = [1,4,12,27,40,42,48,49,52,53,56,58,6,21,34,36,37,41,61,69,73,...
                 5,14,15,20,22,26,29,30,31,32,54,71,79,2,17,23,33,39,57,72,78,80,86,...
                 11,24,63,67,74,81,13,25,47,50,70,75,82,8,18,43,68,76,83,...
                 7,16,35,62,77,84,85,87,88,90,19,44,59,60,64,66,89];

% Store the value of r for each iteration.
r_values = zeros(10000, 1);

% Loop through each row of data.
for i = 1:10000
    % Retrieve the corresponding features in accordance with the re-ordered sequence of questions
    selected_features = zeros(1, 10);
    
    feature_ranges = [1, 13, 23, 32, 45, 55, 61, 68, 74, 84];
    
    for j = 1:length(feature_ranges)-1
        current_range = feature_ranges(j):feature_ranges(j+1)-1;
        selected_features(j) = sum(data(i, current_range));
    end

    % Define the row vector Y.
    Y = [60, 50, 45, 65, 50, 30, 35, 30, 50, 35];

    % Calculate the grey relational coefficient.
    min_diff = min(Y - selected_features);
    max_diff = max(Y - selected_features);
    resolution_coefficient = (min_diff + 0.5 * max_diff) ./ (Y - selected_features + 0.5 * max_diff);

    % Calculate the value of r.
    r_values(i) = 0.0776 * resolution_coefficient(1) + 0.0280 * resolution_coefficient(2) + ...
                  0.0417 * resolution_coefficient(3) + 0.2102 * resolution_coefficient(4) + ...
                  0.1141 * resolution_coefficient(5) + 0.0657 * resolution_coefficient(6) + ...
                  0.0360 * resolution_coefficient(7) + 0.0360 * resolution_coefficient(8) + ...
                  0.2738 * resolution_coefficient(9) + 0.1168 * resolution_coefficient(10);
end

% Draw a scatter plot.
scatter(1:10000, r_values,'o');
xlabel('Sample Number');
ylabel('Grey Relation (r)');
title('Scatter Plot of Grey Relation Coefficients for 10,000 Random Data Points');
grid on;
