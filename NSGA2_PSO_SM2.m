%subspace minimization
function [delayNSGA2_PSO_SM2, energyNSGA2_PSO_SM2] = NSGA2_PSO_SM2(numVehicles, numTasks, initial_pop_size, some_factor, maxIterations, ...
                                           w_delay, w_energy, taskWorkload, taskDeadline, ...
                                           taskStorage, storageCapacity, computingCapacity)

    %% Step 1: Initialize Population using PSO
    population = initializePopulationWithPSO(numVehicles, numTasks, initial_pop_size);

    alpha = struct('position', [], 'fitness', [Inf, Inf]);
    totalDelay = 0;
    totalEnergy = 0;
    delays = zeros(1, maxIterations);
    energies = zeros(1, maxIterations);
    bestFitness = Inf;
    convergenceThreshold = 1e-3;
    startTime = tic;

    %% Step 2: NSGA-II Optimization Loop
    for iter = 1:maxIterations
        prevBestFitness = bestFitness;

        % Evaluate Fitness of Current Population
        for i = 1:length(population)
            vehicleAssignment = max(min(round(population(i).position), numVehicles), 1);
            [totalDelayInd, totalEnergyInd] = evaluateIndividualFitness(vehicleAssignment, ...
                                                                        numTasks, numVehicles, ...
                                                                        computingCapacity, ...
                                                                        taskWorkload, taskDeadline);
            % Ensure fitness is a 2-element vector [delay, energy]
            population(i).fitness = [totalDelayInd, totalEnergyInd];
        end

        % Perform Non-Dominated Sorting
        sortedPopulation = nonDominatedSorting(population);

        % Update alpha
        if sortedPopulation(1).fitness(1) < alpha.fitness(1) && sortedPopulation(1).fitness(2) <= alpha.fitness(2)
            alpha = sortedPopulation(1);
        end

        % Generate next-generation population
        parentPopulation = tournamentSelection(population, initial_pop_size);
        offspringPopulation = crossoverAndMutation(parentPopulation, numVehicles, numTasks, computingCapacity, taskWorkload);
        combinedPopulation = [population; offspringPopulation];
        sortedCombinedPopulation = nonDominatedSorting(combinedPopulation);

        % Refine solutions via subspace minimization
        population = subspaceMinimization(sortedCombinedPopulation, initial_pop_size, numVehicles, numTasks);

        % Update and log results
        bestFitness = alpha.fitness(1) + alpha.fitness(2);
        totalDelay = totalDelay + alpha.fitness(1);
        totalEnergy = totalEnergy + alpha.fitness(2);
        delays(iter) = alpha.fitness(1);
        energies(iter) = alpha.fitness(2);

        % Check for convergence
        if abs(prevBestFitness - bestFitness) / max(1, abs(prevBestFitness)) < convergenceThreshold
            fprintf('Convergence reached at iteration %d\n', iter);
            break;
        end

        % PSO refinement
        population = psoUpdatePopulation(population, numVehicles, numTasks, w_delay, w_energy, taskWorkload, taskDeadline, computingCapacity);
    end

    %% Final Metrics
    executionTime = toc(startTime);
    delayNSGA2_PSO_SM2 = totalDelay / maxIterations;
    energyNSGA2_PSO_SM2 = totalEnergy / maxIterations;

    fprintf('Final delay: %.2f, Final energy: %.2f, Execution time: %.2f seconds\n', ...
            delayNSGA2_PSO_SM2, energyNSGA2_PSO_SM2, executionTime);
end

%% Non-Dominated Sorting Function (Corrected)
function sortedPopulation = nonDominatedSorting(population)
    numIndividuals = length(population);
    front = cell(1, numIndividuals);
    domCount = zeros(1, numIndividuals);
    dominated = cell(1, numIndividuals);

    for p = 1:numIndividuals
        dominated{p} = [];
        domCount(p) = 0;
        for q = 1:numIndividuals
            if dominates(population(p), population(q))
                dominated{p} = [dominated{p}, q];
            elseif dominates(population(q), population(p))
                domCount(p) = domCount(p) + 1;
            end
        end
        if domCount(p) == 0
            front{1} = [front{1}, p];
        end
    end

    i = 1;
    while i <= length(front) && ~isempty(front{i})
        Q = [];
        for p = front{i}
            for q = dominated{p}
                domCount(q) = domCount(q) - 1;
                if domCount(q) == 0
                    Q = [Q, q];
                end
            end
        end
        i = i + 1;
        front{i} = Q;
    end

    % Sort the population by front number
    sortedPopulation = [];
    for i = 1:length(front)
        for j = 1:length(front{i})
            sortedPopulation = [sortedPopulation, population(front{i}(j))];
        end
    end
end

%% Dominance Function (Corrected)
function isDominated = dominates(individual1, individual2)
    isDominated = (individual1.fitness(1) <= individual2.fitness(1) && individual1.fitness(2) < individual2.fitness(2)) || ...
                  (individual1.fitness(1) < individual2.fitness(1) && individual1.fitness(2) <= individual2.fitness(2));
end

%% Subspace Minimization Function
function updatedPopulation = subspaceMinimization(sortedCombinedPopulation, initial_pop_size, numVehicles, numTasks)
    updatedPopulation = sortedCombinedPopulation;

    for i = 1:initial_pop_size
        currentPosition = sortedCombinedPopulation(i).position;
        
        % Check if position size matches numTasks
        if length(currentPosition) ~= numTasks
            error("Mismatch: 'position' should have %d tasks. Found size %d.", numTasks, length(currentPosition));
        end

        % Generate perturbation of the same size as 'position'
        perturbation = rand(size(currentPosition)) * 0.1;  % Small random noise

        % Perform addition
        newPosition = currentPosition + perturbation;

        % Clip 'newPosition' to remain within valid bounds (e.g., between 1 and numVehicles)
        newPosition = max(min(newPosition, numVehicles), 1);

        % Update population
        updatedPopulation(i).position = newPosition;
    end
end

%% PSO Update Function
function population = psoUpdatePopulation(population, numVehicles, numTasks, w_delay, w_energy, taskWorkload, taskDeadline, computingCapacity)
    globalBestPosition = findGlobalBestPosition(population);  % Find the global best position from the current population
    w = 0.5;  % Inertia weight
    c1 = 1.5; % Cognitive coefficient
    c2 = 1.5; % Social coefficient

    for i = 1:length(population)
        velocity = rand(1, numTasks);  % Random velocities
        position = population(i).position;  % Current position of the individual

        % Update position using PSO formula
        newVelocity = w * velocity + c1 * rand() * (globalBestPosition - position) + c2 * rand() * (population(i).position - globalBestPosition);
        population(i).position = position + newVelocity;
        
        % Ensure valid positions (task assignments must be between 1 and numVehicles)
        population(i).position = max(min(population(i).position, numVehicles), 1);
    end
end

%% Tournament Selection Function (Placeholder)
function parentPopulation = tournamentSelection(population, initial_pop_size)
    % Placeholder: Implement tournament selection
    parentPopulation = population(1:initial_pop_size);  % Simplified for now
end

%% Crossover and Mutation Function (Placeholder)
function offspringPopulation = crossoverAndMutation(parentPopulation, numVehicles, numTasks, computingCapacity, taskWorkload)
    % Placeholder: Implement crossover and mutation operations
    offspringPopulation = parentPopulation;  % Simplified for now
end

%% Global Best Position Function (Placeholder)
function globalBestPosition = findGlobalBestPosition(population)
    % Placeholder: Implement logic to find global best position from population
    globalBestPosition = population(1).position;  % Simplified for now
end

%% Example for population initialization
function population = initializePopulationWithPSO(numVehicles, numTasks, initial_pop_size)
    population = struct('position', [], 'fitness', []);
    
    for i = 1:initial_pop_size
        % Ensure that each task is assigned to a valid vehicle index
        population(i).position = randi([1, numVehicles], 1, numTasks);
        population(i).fitness = inf;  % Initialize fitness value
    end
end

%% Fitness Evaluation Function
function [totalDelay, totalEnergy] = evaluateIndividualFitness(vehicleAssignment, ...
                                                              numTasks, numVehicles, ...
                                                              computingCapacity, ...
                                                              taskWorkload, taskDeadline)
    totalDelay = 0;
    totalEnergy = 0;
    
    for task = 1:numTasks
        % Ensure vehicle index is within bounds (1 to numVehicles)
        vehicle = round(vehicleAssignment(task));  
        vehicle = max(min(vehicle, numVehicles), 1);  % Clip the vehicle index between 1 and numVehicles
        
        % Calculate delay based on vehicle's computing capacity
        delay = taskWorkload(task) / computingCapacity(vehicle);  
        
        % Calculate energy consumption (this formula depends on your problem specifics)
        energy = taskWorkload(task) * (taskDeadline(task) / 10);   % Energy consumption
        
        totalDelay = totalDelay + delay;
        totalEnergy = totalEnergy + energy;
    end
end

%% Non-Dominated Sorting Function

%% Dominance Function


% Include all helper functions here, such as:
% - initializePopulationWithPSO
% - evaluateIndividualFitness
% - nonDominatedSorting
% - dominates
% - subspaceMinimization
% - psoUpdatePopulation
% - tournamentSelection (added above)


