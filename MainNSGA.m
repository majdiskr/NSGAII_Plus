%% Step 1: Initialize Parameters
numVehiclesRange = [50,100, 200, 1000];  % Different numbers of vehicles to compare
numTasksRange = [100, 200, 400, 2000];    % Different numbers of tasks to compare
initialPopSizes = [50, 100, 150,200];   % Different population sizes
maxIterations = 100;                % Fixed maximum number of iterations
someFactor = 10;                    % Population size adjustment factor
w_delay = 0.4;                      % Weight for delay
w_energy = 0.6;                     % Weight for energy

% Generate task parameters (using your ranges)
taskWorkload = randi([1, 20], 1, numTasks);       % 1-20 MIPS
taskDeadline = randi([5, 50], 1, numTasks);       % 5-50 seconds
taskStorage = randi([1, 4], 1, numTasks);         % 1-4 GB

% Generate vehicle parameters (using your ranges)
storageCapacity = randi([1, 16], 1, numVehicles); % 1-16 GB
computingCapacity = randi([5, 30], 1, numVehicles); % 5-30 MIPS

% Predefine result storage
results = struct();

%% Step 2: Outer Loop for Parameter Variations
for numVehicles = numVehiclesRange
    for numTasks = numTasksRange
        for initial_pop_size = initialPopSizes
            fprintf('Comparing algorithms for numVehicles=%d, numTasks=%d, initialPopSize=%d\n', ...
                    numVehicles, numTasks, initial_pop_size);
            
           % Generate task parameters
            taskWorkload = randi([1, 5], 1, numTasks) + (0:numTasks-1) * 0.05;  % Decreased task workloads
            taskDeadline = randi([3, 15], 1, numTasks);                         % Decreased task deadlines
            taskStorage = randi([1, 1], 1, numTasks);                           % Reduced task storage range (fixed to 1)
            
            % Generate vehicle parameters
            storageCapacity = randi([1, 4], 1, numVehicles);  % Decreased vehicle storage capacities
            computingCapacity = randi([2, 10], 1, numVehicles);  % Decreased vehicle computing capacities

            % Initialize results for this configuration
            configKey = sprintf('V%d_T%d_P%d', numVehicles, numTasks, initial_pop_size);
            results.(configKey) = struct();

           % Step 3: Call Each Algorithm
            
           % Call and evaluate NSGA2
            fprintf('Running NSGA2...\n');
            resultNSGA2 = NSGA2(numVehicles, numTasks, initial_pop_size, maxIterations, ...
                                w_delay, w_energy, taskWorkload, taskDeadline, ...
                                taskStorage, storageCapacity, computingCapacity);
            
            % Extract delay and energy from the result structure
            delayNSGA2 = resultNSGA2.delay;  % Adjust this field name based on actual implementation
            energyNSGA2 = resultNSGA2.energy;  % Adjust this field name based on actual implementation
            
            % Store results for comparison
            results.(configKey).NSGA2 = struct('delay', delayNSGA2, 'energy', energyNSGA2);

            % Call and evaluate NSGA3
            fprintf('Running NSGA3...\n');
            resultNSGA3 = NSGA3(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                w_delay, w_energy, taskWorkload, taskDeadline, ...
                                taskStorage, storageCapacity, computingCapacity);
            
            % Extract fields from the result structure
            delayNSGA3 = resultNSGA3.delay;  % Adjust field name if necessary
            energyNSGA3 = resultNSGA3.energy;  % Adjust field name if necessary
            
            % Store results for comparison
            results.(configKey).NSGA3 = struct('delay', delayNSGA3, 'energy', energyNSGA3);

            
            % % Call and evaluate NSGA2+
            fprintf('Running NSGA2P...\n');
            [delayNSGA2P, energyNSGA2P] = NSGA2_2(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                                  w_delay, w_energy, taskWorkload, taskDeadline, ...
                                                  taskStorage, storageCapacity, computingCapacity);
            results.(configKey).NSGA2P = struct('delay', delayNSGA2P, 'energy', energyNSGA2P);


             % % Call and evaluate NSGA2++
            fprintf('Running NSGA2P...\n');
            [delayNSGA2Plus, energyNSGA2Plus] = NSGA2_3(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                                  w_delay, w_energy, taskWorkload, taskDeadline, ...
                                                  taskStorage, storageCapacity, computingCapacity);
            results.(configKey).NSGA2Plus = struct('delay', delayNSGA2Plus, 'energy', energyNSGA2Plus);


             % Call and evaluate NSGA2_PSO
            fprintf('Running NSGA2_PSO...\n');
            [delayNSGA2_PSO, energyNSGA2_PSO] = NSGA2_PSO(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                                    w_delay, w_energy, taskWorkload, taskDeadline, ...
                                                    taskStorage, storageCapacity, computingCapacity);
            results.(configKey).NSGA2_PSO = struct('delay', delayNSGA2_PSO, 'energy', energyNSGA2_PSO);

             % Call and evaluate NSGA2_PSO_SM
            fprintf('Running NSGA2_PSO...\n');
            [delayNSGA2_PSO_SM, energyNSGA2_PSO_SM] = NSGA2_PSO_SM(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                                    w_delay, w_energy, taskWorkload, taskDeadline, ...
                                                    taskStorage, storageCapacity, computingCapacity);
            results.(configKey).NSGA2_PSO_SM = struct('delay', delayNSGA2_PSO_SM, 'energy', energyNSGA2_PSO_SM);



        end
    end
end

%% Step 4: Analyze and Compare Results
% Extract results for analysis and visualization
fields = fieldnames(results);
for i = 1:numel(fields)
    configKey = fields{i};
    fprintf('Results for %s:\n', configKey);
    disp(results.(configKey));  % Display results for each configuration
end

% Optional: Visualize results with bar plots or other figures
% Example:
% plotComparison(results, numVehiclesRange, numTasksRange, initialPopSizes);

%% Step 5: Visualization of Results
% Initialize figure
figure;

% Prepare data for visualization
algorithms1 = {'NSGA2P','NSGA2_PSO', 'NSGA2_PSO_SM'};
algorithms = {'NSGA2Plus', 'NSGA2P','NSGA2', 'NSGA3','NSGA2_PSO', 'NSGA2_PSO_SM'};

delayData = [];
energyData = [];
labels = {};

for i = 1:numel(fields)
    configKey = fields{i};
    labels{end+1} = configKey;
    
    % Extract delay and energy for each algorithm
    for algo = 1:numel(algorithms)
        algorithmName = algorithms{algo};
        if isfield(results.(configKey), algorithmName)
            delayData(i, algo) = results.(configKey).(algorithmName).delay;
            energyData(i, algo) = results.(configKey).(algorithmName).energy;
        else
            delayData(i, algo) = NaN; % Handle missing data
            energyData(i, algo) = NaN;
        end
    end
end

% Plot delay comparison
subplot(2, 1, 1);
bar(delayData, 'grouped');
set(gca, 'XTickLabel', labels);
title('Comparison of Delay Across Algorithms');
xlabel('Configuration');
ylabel('Delay');
legend(algorithms, 'Location', 'northeastoutside');
grid on;

% Plot energy comparison
subplot(2, 1, 2);
bar(energyData, 'grouped');
set(gca, 'XTickLabel', labels);
title('Comparison of Energy Across Algorithms');
xlabel('Configuration');
ylabel('Energy');
legend(algorithms, 'Location', 'northeastoutside');
grid on;

% Adjust figure properties
set(gcf, 'Position', [100, 100, 1200, 800]);

%% Save Figures
saveas(gcf, 'Algorithm_Comparison_Charts.png');
