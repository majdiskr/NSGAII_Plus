function NSGA3(numVehicles, numTasks, initial_pop_size, some_factor, maxIterations, ...
                 w_delay, w_energy, taskWorkload, taskDeadline, ...
                 taskStorage, storageCapacity, computingCapacity)

    %% Step 1: Initialize Population
    population = initializePopulation(initial_pop_size, numVehicles, numTasks);
    alpha = struct('position', [], 'fitness', [Inf, Inf]);
    totalDelay = 0;  % Accumulator for delay
    totalEnergy = 0; % Accumulator for energy
    delays = zeros(1, maxIterations);
    energies = zeros(1, maxIterations);
    bestFitness = Inf; % Initialize best fitness for convergence check
    convergenceThreshold = 1e-3;
    startTime = tic; % Start timer

    %% Step 2: NSGA-III Optimization Loop
    % Define reference points based on number of objectives
    numObjectives = 2; % Delay and Energy
    referencePoints = generateReferencePoints(numObjectives, initial_pop_size);
    
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
        offspringPopulation = crossoverAndMutation(parentPopulation, numVehicles, numTasks);
        combinedPopulation = [population; offspringPopulation];
        sortedCombinedPopulation = nonDominatedSorting(combinedPopulation);

        % Apply NSGA-III Selection using reference points
        population = selectNSGAIII(sortedCombinedPopulation, initial_pop_size, referencePoints);

        % Update Best Fitness and Accumulate Results
        bestFitness = alpha.fitness(1) + alpha.fitness(2);
        totalDelay = totalDelay + alpha.fitness(1);
        totalEnergy = totalEnergy + alpha.fitness(2);

        % Display Progress
        fprintf('Iteration %d: Best Fitness (Delay + Energy): %.2f\n', iter, bestFitness);

        % Check for Convergence
        if abs(prevBestFitness - bestFitness) < convergenceThreshold
            fprintf('Convergence reached at iteration %d\n', iter);
            break;
        end
    end

    %% Output Results
    executionTime = toc(startTime); % End timer
    averageEnergy = totalEnergy / maxIterations; % Calculate average energy
    averageDelay = totalDelay / maxIterations;   % Calculate average delay
    stdDevEnergy = std(energies);
    stdDevDelay = std(delays);

    fprintf('totalEnergy: %.2f and maxIterations: %.2f\n', totalEnergy, maxIterations);
    fprintf('Best solution found with Delay_NSAGIII: %.2f and Energy: %.2f\n', alpha.fitness(1), alpha.fitness(2));
    fprintf('Average Energy Consumption_NSAGIII: %.2f\n', averageEnergy);
    fprintf('Average Delay_NSAGIII: %.2f\n', averageDelay);
    fprintf('Standard Deviation of Energy NSGAIII: %.2f\n', stdDevEnergy);
    fprintf('Standard Deviation of Delay NSGAIII: %.2f\n', stdDevDelay);
    fprintf('Total Execution Time NSGAIII: %.2f seconds\n', executionTime);
end

%% Helper Functions

function referencePoints = generateReferencePoints(numObjectives, popSize)
    % Generate reference points for NSGA-III (uniformly distributed)
    referencePoints = lhsdesign(popSize, numObjectives); % Latin Hypercube Sampling for uniform distribution
end

function population = selectNSGAIII(population, popSize, referencePoints)
    % Perform selection using NSGA-III approach with reference points
    % First, assign reference points to individuals
    for i = 1:length(population)
        population(i).refPointDistance = calculateReferencePointDistance(population(i), referencePoints);
    end
    
    % Sort by reference point distance and select the top individuals
    [~, sortedIdx] = sort([population.refPointDistance]);
    population = population(sortedIdx(1:popSize)); % Select the top individuals based on reference points
end

function distance = calculateReferencePointDistance(individual, referencePoints)
    % Calculate the Euclidean distance between individual and each reference point
    fitness = individual.fitness;
    distances = sqrt(sum((referencePoints - fitness).^2, 2)); % Euclidean distance
    [~, minIdx] = min(distances);
    distance = distances(minIdx);
end

% Other functions (initializePopulation, evaluateIndividualFitness, etc.) remain unchanged

%% Helper Functions

function population = initializePopulation(popSize, numVehicles, numTasks)
    % Initialize the population with random assignments
    population = struct();
    for i = 1:popSize
        population(i).position = randi([1, numVehicles], 1, numTasks);
        population(i).fitness = [0, 0]; % Placeholder for [delay, energy]
        population(i).crowdingDistance = 0; % Initialize crowding distance
    end
end

function population = evaluatePopulationFitness(population, numTasks, numVehicles, ...
                                                computingCapacity, taskWorkload, ...
                                                taskDeadline)
    % Evaluate the fitness (delay and energy) of the population
    for i = 1:length(population)
        vehicleAssignment = max(min(population(i).position, numVehicles), 1);
        [totalDelay, totalEnergy] = evaluateIndividualFitness(vehicleAssignment, ...
                                                              numTasks, numVehicles, ...
                                                              computingCapacity, ...
                                                              taskWorkload, taskDeadline);
        population(i).fitness = [totalDelay, totalEnergy];
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
        vehicle = vehicleAssignment(task);
        delay = taskWorkload(task) / computingCapacity(vehicle);
        energy = taskWorkload(task) * (taskDeadline(task) / 10);
        totalDelay = totalDelay + delay;
        totalEnergy = totalEnergy + energy;
    end
end

function selectedParents = tournamentSelection(population, numParents)
    % Perform tournament selection
    selectedParents = repmat(population(1), 1, numParents);
    for i = 1:numParents
        idx = randperm(length(population), 2);
        individual1 = population(idx(1));
        individual2 = population(idx(2));
        if dominates(individual1, individual2)
            selectedParents(i) = individual1;
        else
            selectedParents(i) = individual2;
        end
    end
end

function offspringPopulation = crossoverAndMutation(parents, numVehicles, numTasks)
    % Perform crossover and mutation
    offspringPopulation = parents;
    for i = 1:length(parents)
        parent1 = parents(randi(length(parents)));
        parent2 = parents(randi(length(parents)));
        crossoverPoint = randi(numTasks - 1);
        offspringPopulation(i).position = ...
            [parent1.position(1:crossoverPoint), parent2.position(crossoverPoint+1:end)];
        mutationRate = 0.1;
        for j = 1:numTasks
            if rand < mutationRate
                offspringPopulation(i).position(j) = randi(numVehicles);
            end
        end
    end
end

function sortedPopulation = nonDominatedSorting(population)
    % Perform non-dominated sorting
    % Add logic to classify individuals into Pareto fronts
    sortedPopulation = population; % Placeholder
end

function population = calculateCrowdingDistance(population)
    % Compute crowding distances for sorting
    for i = 1:length(population)
        population(i).crowdingDistance = rand(); % Placeholder logic
    end
end

function nextGeneration = selectNextGeneration(population, popSize)
    % Select the next generation based on non-domination and crowding distance
    nextGeneration = population(1:popSize); % Placeholder logic
end

function result = dominates(ind1, ind2)
    % Check if ind1 dominates ind2
    result = all(ind1.fitness <= ind2.fitness) && any(ind1.fitness < ind2.fitness);
end
