function [delayNSGA2_2, energyNSGA2_2] = NSGA2_2(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                          w_delay, w_energy, taskWorkload, taskDeadline, ...
                                          taskStorage, storageCapacity, computingCapacity);


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
        offspringPopulation = crossoverAndMutation(parentPopulation, numVehicles, numTasks);
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
        fprintf('Iteration %d: Best Fitness (Delay + Energy) NSGA2_2: %.2f\n', iter, bestFitness);

        % Check for Convergence
        if abs(prevBestFitness - bestFitness) < convergenceThreshold
            fprintf('Convergence reached at iteration NSGA2_2 %d\n', iter);
            break;
        end
    end

    %% Output Results
    executionTime = toc(startTime); % End timer
    averageEnergy = totalEnergy / maxIterations; % Calculate average energy
    averageDelay = totalDelay / maxIterations;   % Calculate average delay
    stdDevEnergy = std(energies);
    stdDevDelay = std(delays);

    fprintf('Best solution found with Delay_NSGA2_2: %.2f and Energy: %.2f\n', alpha.fitness(1), alpha.fitness(2));
    fprintf('Average Energy Consumption_NSGA2_2: %.2f\n', averageEnergy);
    fprintf('Average Delay_NSGA2_2: %.2f\n', averageDelay);
    fprintf('Standard Deviation of Energy NSGA2_2: %.2f\n', stdDevEnergy);
    fprintf('Standard Deviation of Delay NSGA2_2: %.2f\n', stdDevDelay);
    fprintf('Total Execution Time NSGA2_2: %.2f seconds\n', executionTime);
    
    % Return delay and energy for Main2
    delayNSGA2_2 = averageDelay;
    energyNSGA2_2 = averageEnergy;

end

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
        % Sort the individuals by the m-th objective
        [~, idx] = sort([population.fitness], m);
        population(idx(1)).crowdingDistance = Inf;
        population(idx(end)).crowdingDistance = Inf;

        for i = 2:numIndividuals-1
            if ~isempty(population(idx(i+1))) && ~isempty(population(idx(i-1)))
                population(idx(i)).crowdingDistance = population(idx(i)).crowdingDistance + ...
                    (population(idx(i+1)).fitness(m) - population(idx(i-1)).fitness(m));
            end
        end
    end
end

function nextGeneration = selectNextGeneration(population, popSize)
    % Select the next generation based on non-domination and crowding distance
    nextGeneration = population(1:popSize);
end

function result = dominates(ind1, ind2)
    % Check if ind1 dominates ind2
    result = all(ind1.fitness <= ind2.fitness) && any(ind1.fitness < ind2.fitness);
end
