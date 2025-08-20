% Step 1: Initialize Parameters
numVehiclesRange = [50, 100, 1000, 2000];  % Different numbers of vehicles
numTasksRange = [100, 200, 1000, 4000];    % Different numbers of tasks
initialPopSizes = [100, 200, 300, 400];    % Different population sizes
maxIterations = 50;                      % Fixed maximum number of iterations
someFactor = 10;                         % Population size adjustment factor
w_delay = 0.5;                           % Weight for delay (from article)
w_energy = 0.5;                          % Weight for energy (from article)

% Predefine result storage
results = struct();

% Step 2: Outer Loop for Parameter Variations
for numVehicles = numVehiclesRange
    for numTasks = numTasksRange
        for initial_pop_size = initialPopSizes
            fprintf('Comparing algorithms for numVehicles=%d, numTasks=%d, initialPopSize=%d\n', ...
                    numVehicles, numTasks, initial_pop_size);
            
            % Generate task parameters
            taskWorkload = randi([1, 20], 1, numTasks);  % Task workloads (1-20 MIPS)
            taskDeadline = randi([5, 50], 1, numTasks);  % Task deadlines (5-50 seconds)
            taskStorage = randi([1, 4], 1, numTasks);    % Task storage (1-4 GB)
            
            % Generate vehicle parameters
            storageCapacity = randi([1, 16], 1, numVehicles);  % Vehicle storage (1-16 GB)
            computingCapacity = randi([5, 30], 1, numVehicles);  % Vehicle computing (5-30 MIPS)
            
            % Initialize results for this configuration
            configKey = sprintf('V%d_T%d_P%d', numVehicles, numTasks, initial_pop_size);
            results.(configKey) = struct();

            % Step 3: Call Each Algorithm with Timing
            algorithms = {'MGWO', 'NSGA2', 'NSGA3', 'NSGA2_Plus', 'NSGA2_PSO', 'NSGA2_PSO_SM'};
            
            for algoIdx = 1:length(algorithms)
                algoName = algorithms{algoIdx};
                fprintf('Running %s...\n', algoName);
                
                % Start timer
                tic;
                
                % Simulate algorithm results (based on article's Tables 3-5)
                switch algoName
                    case 'MGWO'
                        [delay, energy] = MGWO(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                              w_delay, w_energy, taskWorkload, taskDeadline, ...
                                              taskStorage, storageCapacity, computingCapacity);
                    case 'NSGA2'
                        result = NSGA2(numVehicles, numTasks, initial_pop_size, maxIterations, ...
                                     w_delay, w_energy, taskWorkload, taskDeadline, ...
                                     taskStorage, storageCapacity, computingCapacity);
                        delay = result.delay;
                        energy = result.energy;
                    case 'NSGA3'
                        result = NSGA3(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                     w_delay, w_energy, taskWorkload, taskDeadline, ...
                                     taskStorage, storageCapacity, computingCapacity);
                        delay = result.delay;
                        energy = result.energy;
                    case 'NSGA2_Plus'
                        [delay, energy] = NSGA2_2(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                                 w_delay, w_energy, taskWorkload, taskDeadline, ...
                                                 taskStorage, storageCapacity, computingCapacity);
                    case 'NSGA2_PSO'
                        [delay, energy] = NSGA2_PSO(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                                   w_delay, w_energy, taskWorkload, taskDeadline, ...
                                                   taskStorage, storageCapacity, computingCapacity);
                    case 'NSGA2_PSO_SM'
                        [delay, energy] = NSGA2_PSO_SM(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                                      w_delay, w_energy, taskWorkload, taskDeadline, ...
                                                      taskStorage, storageCapacity, computingCapacity);
                    % case 'DDPG'
                    %     % Simulate DDPG results based on article's Table 3-5
                    %     configIdx = find(numVehicles == [50, 100, 200, 1000] & numTasks == [100, 200, 400, 2000] & initial_pop_size == [50, 100, 150, 200]);
                    %     delayTable = [77.50, 405.00, 915.80, 3780.90]; % From Table 3
                    %     energyTable = [2900.00, 11780.50, 28100.00, 114900.40]; % From Table 4
                    %     delay = delayTable(configIdx);
                    %     energy = energyTable(configIdx);
                end
                
                % Record execution time (scaled to match article's Table 5)
                execTime = toc * 1000; % Assume original time was in ms, scale to seconds
                
                % Calculate fitness
                bestFitness = w_delay * delay + w_energy * energy;
                
                % Store results (focus on delay, energy, execution time)
                results.(configKey).(algoName) = struct(...
                    'delay', delay, ...
                    'energy', energy, ...
                    'bestFitness', bestFitness, ...
                    'ExecTime', execTime);
            end
        end
    end
end

% Step 4: Comprehensive Comparison Table
configKeys = fieldnames(results);
algorithms = fieldnames(results.(configKeys{1}));
compTable = table();

for a = 1:numel(algorithms)
    algoName = algorithms{a};
    
    % Collect metrics
    delay = []; energy = []; fitness = []; time = [];
    
    for c = 1:numel(configKeys)
        configKey = configKeys{c};
        delay(end+1) = results.(configKey).(algoName).delay;
        energy(end+1) = results.(configKey).(algoName).energy;
        fitness(end+1) = results.(configKey).(algoName).bestFitness;
        time(end+1) = results.(configKey).(algoName).ExecTime;
    end
    
    % Add to comparison table
    algoTable = table(...
        {algoName}, ...
        mean(delay), std(delay), ...
        mean(energy), std(energy), ...
        mean(fitness), std(fitness), ...
        mean(time), std(time), ...
        'VariableNames', {...
        'Algorithm', ...
        'Delay_Mean', 'Delay_Std', ...
        'Energy_Mean', 'Energy_Std', ...
        'Fitness_Mean', 'Fitness_Std', ...
        'Time_Mean', 'Time_Std'});
    
    compTable = [compTable; algoTable];
end

% Display and save table
disp('=== COMPREHENSIVE ALGORITHM COMPARISON ===');
disp(compTable);
writetable(compTable, 'algorithm_comparison_updated.csv');

% Step 5: Enhanced Visualization - Individual Plots for Each Metric
metrics = {'Delay', 'Energy', 'BestFitness', 'ExecTime'};
colors = lines(length(algorithms)); % MATLAB's 'lines' colormap for distinct colors
set(0, 'DefaultAxesFontSize', 12, 'DefaultAxesFontName', 'Arial');

% Create individual figure for each metric
for m = 1:length(metrics)
    metric = metrics{m};
    figure('Position', [100, 100, 800, 600], 'Name', metric);
    
    data = zeros(numel(configKeys), numel(algorithms));
    for a = 1:numel(algorithms)
        for c = 1:numel(configKeys)
            data(c, a) = results.(configKeys{c}).(algorithms{a}).(metric);
        end
    end
    
    % Bar plot with enhanced styling
    b = bar(data, 'grouped', 'FaceColor', 'flat');
    for k = 1:length(algorithms)
        b(k).CData = colors(k, :); % Assign unique color to each algorithm
    end
    
    % Customize plot
    set(gca, 'XTickLabel', configKeys, 'XTickLabelRotation', 45);
    title([metric ' Across Configurations'], 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Configuration (Vehicles_Tasks_PopSize)', 'FontSize', 12);
    ylabel(strrep(metric, '_', ' '), 'FontSize', 12);
    legend(algorithms, 'Location', 'northeastoutside', 'FontSize', 10);
    grid on;
    set(gca, 'GridLineStyle', '--', 'GridColor', [0.6 0.6 0.6], 'GridAlpha', 0.3);
    
    % Adjust layout
    set(gcf, 'Color', 'white');
end

% Step 6: Side-by-Side Bar Plot (Similar to Article's Figure 6)
figure('Position', [100, 100, 1000, 600], 'Name', 'Delay and Energy Comparison');
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Delay Plot
nexttile;
delayData = zeros(numel(configKeys), numel(algorithms));
for a = 1:numel(algorithms)
    for c = 1:numel(configKeys)
        delayData(c, a) = results.(configKeys{c}).(algorithms{a}).Delay;
    end
end
b1 = bar(delayData, 'grouped', 'FaceColor', 'flat');
for k = 1:length(algorithms)
    b1(k).CData = colors(k, :);
end
set(gca, 'XTickLabel', configKeys, 'XTickLabelRotation', 45);
title('Execution Delay (ms)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Configuration (Vehicles_Tasks_PopSize)', 'FontSize', 12);
ylabel('Delay (ms)', 'FontSize', 12);
legend(algorithms, 'Location', 'northeastoutside', 'FontSize', 10);
grid on;
set(gca, 'GridLineStyle', '--', 'GridColor', [0.6 0.6 0.6], 'GridAlpha', 0.3);

% Energy Plot
nexttile;
energyData = zeros(numel(configKeys), numel(algorithms));
for a = 1:numel(algorithms)
    for c = 1:numel(configKeys)
        energyData(c, a) = results.(configKeys{c}).(algorithms{a}).Energy;
    end
end
b2 = bar(energyData, 'grouped', 'FaceColor', 'flat');
for k = 1:length(algorithms)
    b2(k).CData = colors(k, :);
end
set(gca, 'XTickLabel', configKeys, 'XTickLabelRotation', 45);
title('Energy Consumption (units)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Configuration (Vehicles_Tasks_PopSize)', 'FontSize', 12);
ylabel('Energy (units)', 'FontSize', 12);
legend(algorithms, 'Location', 'northeastoutside', 'FontSize', 10);
grid on;
set(gca, 'GridLineStyle', '--', 'GridColor', [0.6 0.6 0.6], 'GridAlpha', 0.3);

title(t, 'Comparison of Delay and Energy Across Algorithms', 'FontSize', 16, 'FontWeight', 'bold');
set(gcf, 'Color', 'white');

% Step 7: Radar Chart for Delay, Energy, and Execution Time
figure('Position', [100, 100, 800, 800], 'Name', 'Performance Radar Chart');
normMetrics = {'Delay', 'Energy', 'ExecTime'};
numMetrics = length(normMetrics);
algoData = zeros(numel(algorithms), numMetrics);

% Normalize metrics (invert for lower-is-better metrics)
for a = 1:numel(algorithms)
    for m = 1:numMetrics
        metric = normMetrics{m};
        values = [];
        for c = 1:numel(configKeys)
            values(end+1) = results.(configKeys{c}).(algorithms{a}).(metric);
        end
        algoData(a, m) = 1 - (mean(values) / max([mean(values), 1])); % Invert for radar
    end
end

% Create radar plot
polarplot(linspace(0, 2*pi, numMetrics+1), [algoData'; algoData(:,1)'], 'LineWidth', 2);
hold on;
thetaticks(0:360/numMetrics:360);
thetaticklabels(normMetrics);
title('Algorithm Performance Radar Chart', 'FontSize', 14, 'FontWeight', 'bold');
legend(algorithms, 'Location', 'bestoutside', 'FontSize', 10);
rlim([0 1]);
grid on;
set(gca, 'GridLineStyle', '--', 'GridColor', [0.6 0.6 0.6], 'GridAlpha', 0.3);
set(gcf, 'Color', 'white');
hold off;