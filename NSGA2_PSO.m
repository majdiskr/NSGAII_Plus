function [delayNSGA2_PSO, energyNSGA2_PSO] = NSGA2_PSO(numVehicles, numTasks, initial_pop_size, some_factor, maxIterations, ...
                                           w_delay, w_energy, taskWorkload, taskDeadline, ...
                                           taskStorage, storageCapacity, computingCapacity)

    %% Step 1: Initialize Population using PSO
    population = initializePopulationWithPSO(initial_pop_size, numVehicles, numTasks, w_delay, w_energy, taskWorkload, taskDeadline, computingCapacity);
    alpha = struct('position', [], 'fitness', [Inf, Inf]);
    totalDelay = 0;  % Accumulator for delay delay = taskWorkload(task) / computingCapacity(vehicle);
    totalEnergy = 0; % Accumulator for energy
    delays = zeros(1, maxIterations);
    energies = zeros(1, maxIterations);
    bestFitness = Inf; % Initialize best fitness for convergence check
    convergenceThreshold = 1e-3;
    startTime = tic; % Start timer

    %% Step 2: NSGA-II Optimization Loop
    for iter = 1:maxIterations
        prevBestFitness = bestFitness;

        % Evaluate Fitness of Current Population
        for i = 1:length(population)
            vehicleAssignment = max(min(population(i).position, numVehicles), 1); % Ensure valid indices
            [totalDelayInd, totalEnergyInd] = evaluateIndividualFitness(vehicleAssignment, ...
                                                                        numTasks, numVehicles, ...
                                                                        computingCapacity, ...
                                                                        taskWorkload, taskDeadline);
            population(i).fitness = [totalDelayInd, totalEnergyInd];
        end

        % Perform Non-Dominated Sorting
        sortedPopulation = nonDominatedSorting(population);

        % Update alpha solution (best Pareto-dominant solution)
        if sortedPopulation(1).fitness(1) < alpha.fitness(1) && sortedPopulation(1).fitness(2) <= alpha.fitness(2)
            alpha = sortedPopulation(1);
        end

        % Combine Parent and Offspring Population for Next Generation
        parentPopulation = tournamentSelection(population, initial_pop_size);
        offspringPopulation = crossoverAndMutation(parentPopulation, numVehicles, numTasks, computingCapacity, taskWorkload);
        combinedPopulation = [population; offspringPopulation];
        sortedCombinedPopulation = nonDominatedSorting(combinedPopulation);
        population = selectNextGeneration(sortedCombinedPopulation, initial_pop_size);

        % Update Best Fitness and Accumulate Results
        bestFitness = alpha.fitness(1) + alpha.fitness(2);
        totalDelay = totalDelay + alpha.fitness(1);
        totalEnergy = totalEnergy + alpha.fitness(2);
        delays(iter) = alpha.fitness(1);
        energies(iter) = alpha.fitness(2);

        % Display Progress
        fprintf('Iteration %d: Best Fitness (Delay + Energy) NSGA2_PSO: %.2f\n', iter, bestFitness);

        % Check for Convergence
        if abs(prevBestFitness - bestFitness) < convergenceThreshold
            fprintf('Convergence reached at iteration NSGA2_PSO %d\n', iter);
            break;
        end

        %% Step 3: Use PSO to refine solutions for Energy Efficiency
        % Apply PSO to improve the population towards energy minimization
        population = psoUpdatePopulation(population, numVehicles, numTasks, w_delay, w_energy, taskWorkload, taskDeadline, computingCapacity);
    end

    %% Output Results
    executionTime = toc(startTime); % End timer
    averageEnergy = totalEnergy / maxIterations; % Calculate average energy
    averageDelay = totalDelay / maxIterations;   % Calculate average delay
    stdDevEnergy = std(energies);
    stdDevDelay = std(delays);

    fprintf('Best solution found with Delay_NSGA2_PSO: %.2f and Energy: %.2f\n', alpha.fitness(1), alpha.fitness(2));
    fprintf('Average Energy Consumption_NSGA2_PSO: %.2f\n', averageEnergy);
    fprintf('Average Delay_NSGA2_PSO: %.2f\n', averageDelay);
    fprintf('Standard Deviation of Energy NSGA2_PSO: %.2f\n', stdDevEnergy);
    fprintf('Standard Deviation of Delay NSGA2_PSO: %.2f\n', stdDevDelay);
    fprintf('Total Execution Time NSGA2_PSO: %.2f seconds\n', executionTime);
    
    % Return delay and energy for Main2
    delayNSGA2_PSO = averageDelay;
    energyNSGA2_PSO = averageEnergy;
end

%% Step 4: PSO Update Function for Energy Minimization
function population = psoUpdatePopulation(population, numVehicles, numTasks, w_delay, w_energy, taskWorkload, taskDeadline, computingCapacity)
    % PSO update for energy minimization
    globalBestPosition = findGlobalBestPosition(population);  % Find the global best position from the current population
    w = 0.5; % Inertia weight
    c1 = 1.5; % Cognitive coefficient
    c2 = 1.5; % Social coefficient

    for i = 1:length(population)
        % Update velocities and positions
        velocity = rand(1, numTasks);  % Random velocities
        position = population(i).position;  % Current position of the individual

        % Update position using PSO formula
        newVelocity = w * velocity + c1 * rand() * (globalBestPosition - position) + c2 * rand() * (population(i).position - globalBestPosition);
        population(i).position = position + newVelocity;
        
        % Ensure valid positions (e.g., task assignments must be between 1 and numVehicles)
        population(i).position = max(min(population(i).position, numVehicles), 1);
    end
end

%% Helper Functions (Non-Dominated Sorting, Selection, and Crossover)

function globalBestPosition = findGlobalBestPosition(population)
    % Find the global best position from the population
    globalBestPosition = population(1).position;  % Start with first position
    globalBestFitness = population(1).fitness(2);  % Start with first individual's energy
    for i = 2:length(population)
        if population(i).fitness(2) < globalBestFitness  % Minimize energy
            globalBestPosition = population(i).position;
            globalBestFitness = population(i).fitness(2);
        end
    end
end

function [totalDelay, totalEnergy] = evaluateIndividualFitness(vehicleAssignment, ...
                                                              numTasks, numVehicles, ...
                                                              computingCapacity, ...
                                                              taskWorkload, taskDeadline)
    % Compute delay and energy for a given task assignment
    totalDelay = 0;
    totalEnergy = 0;
    
    for task = 1:numTasks
        % Ensure valid vehicle index for each task (vehicle assignment must be between 1 and numVehicles)
        vehicle = round(vehicleAssignment(task));  % Round to nearest integer for valid vehicle index
        vehicle = max(min(vehicle, numVehicles), 1);  % Ensure vehicle is within bounds 1 <= vehicle <= numVehicles
        
        % Calculate delay based on vehicle's computing capacity
        if vehicle <= numVehicles  % Ensure vehicle is within valid bounds
            delay = taskWorkload(task) / computingCapacity(vehicle);  
        else
            error('Invalid vehicle index in the assignment. Please check the vehicle assignments.');
        end
        
        energy = taskWorkload(task) * (taskDeadline(task) / 10);   % Energy consumption
        
        totalDelay = totalDelay + delay;
        totalEnergy = totalEnergy + energy;
    end
end

function sortedPopulation = nonDominatedSorting(population)
    % Perform non-dominated sorting
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
        if ~isempty(Q)
            front{i} = Q;
        end
    end

    % Combine all fronts into sorted population
    sortedPopulation = population([front{:}]);
end

function population = calculateCrowdingDistance(population)
    % Compute crowding distances for sorting
    numIndividuals = length(population);
    if numIndividuals < 3
        for i = 1:numIndividuals
            population(i).crowdingDistance = Inf;
        end
        return;
    end

    for m = 1:2 % Assuming two objectives: delay and energy
        [~, idx] = sort([population.fitness], m);
        population(idx(1)).crowdingDistance = Inf;
        population(idx(end)).crowdingDistance = Inf;

        % Calculate crowding distance for other individuals
        for i = 2:numIndividuals - 1
            population(idx(i)).crowdingDistance = population(idx(i)).crowdingDistance + ...
                (population(idx(i + 1)).fitness(m) - population(idx(i - 1)).fitness(m));
        end
    end
end

%% Step 4: PSO Update Function for Energy Minimization

%% Helper Functions (Non-Dominated Sorting, Selection, and Crossover)



% Other helper functions

%% Step 4: PSO Update Function for Energy Minimization

%% Step 5: Helper Functions


function selectedPopulation = tournamentSelection(population, numParents)
    % Perform tournament selection for reproduction
    selectedPopulation = [];
    popSize = length(population);
    
    for i = 1:numParents
        % Ensure we always have at least 2 individuals for the tournament
        if popSize < 2
            error('Not enough individuals in population for tournament selection.');
        end
        
        % Select two individuals randomly for the tournament
        idx = randi(popSize, [1, 2]); % Choose two indices within the population size
        
        % Ensure indices are within bounds
        idx = mod(idx - 1, popSize) + 1; % Wrap around to valid indices (1 to popSize)

        % Compare their fitness and select the best one
        [~, bestIdx] = min([population(idx(1)).fitness(1) + population(idx(1)).fitness(2), ...
                            population(idx(2)).fitness(1) + population(idx(2)).fitness(2)]);  % Choose the best
        
        % Append the selected individual to the population
        selectedPopulation = [selectedPopulation, population(idx(bestIdx))];
    end
end

function selectedPopulation = selectNextGeneration(sortedPopulation, popSize)
    % Select the next generation
    selectedPopulation = sortedPopulation(1:popSize);
end

function isDom = dominates(ind1, ind2)
    % Check if ind1 dominates ind2
    isDom = all(ind1.fitness <= ind2.fitness) && any(ind1.fitness < ind2.fitness);
end

%% Step 4: PSO Update Function for Energy Minimization

%% Step 5: Helper Functions


% Helper functions for sorting, selection, and mutation are the same as provided above

%% Step 4: PSO Update Function for Energy Minimization

%% Step 5: Helper Functions


function offspringPopulation = crossoverAndMutation(parents, numVehicles, numTasks, computingCapacity, taskWorkload)
    % Perform crossover and mutation for the offspring population
    offspringPopulation = parents;
    
    for i = 1:length(parents)
        parent1 = parents(randi(length(parents)));
        parent2 = parents(randi(length(parents)));
        
        % Crossover: Create a new offspring by combining two parents
        crossoverPoint = randi(numTasks - 1);
        offspringPopulation(i).position = ...
            [parent1.position(1:crossoverPoint), parent2.position(crossoverPoint+1:end)];
        
        % Mutation: Randomly change task assignments with some probability
        mutationRate = 0.1; % Set mutation rate
        for j = 1:numTasks
            if rand < mutationRate
                offspringPopulation(i).position(j) = randi(numVehicles);  % Assign a new vehicle
            end
        end
        
        % Make sure all assignments are valid
        offspringPopulation(i).position = max(min(offspringPopulation(i).position, numVehicles), 1);
    end
end

function population = assignTasksToVehicles(numVehicles, numTasks, computingCapacity, taskWorkload)
    % Assign tasks to vehicles based on computing capacity and task workload
    population = struct();
    for i = 1:numVehicles
        % Randomly assign tasks to vehicles
        population(i).position = randi([1, numVehicles], 1, numTasks);
        population(i).fitness = [0, 0]; % Placeholder for [delay, energy]
    end
end

% Updated initialization for PSO
function population = initializePopulationWithPSO(popSize, numVehicles, numTasks, w_delay, w_energy, taskWorkload, taskDeadline, computingCapacity)
    % Initialize the population using PSO initialization logic
    population = struct();
    for i = 1:popSize
        population(i).position = randi([1, numVehicles], 1, numTasks);  % Random initial positions
        population(i).fitness = [0, 0]; % Placeholder for [delay, energy]
        population(i).crowdingDistance = 0; % Initialize crowding distance
    end
end
