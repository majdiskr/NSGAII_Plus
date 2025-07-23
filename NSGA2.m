function result = NSGA2(numVehicles, numTasks, initial_pop_size, maxIterations, ...
                  w_delay, w_energy, taskWorkload, taskDeadline, ...
                  taskStorage, storageCapacity, computingCapacity)

    %% Step 1: Initialize Population
    population = initializePopulation(initial_pop_size, numVehicles, numTasks);
    popSize = initial_pop_size;

    % Initialize metrics
    totalDelay = 0;
    totalEnergy = 0;
    delays = zeros(1, maxIterations);
    energies = zeros(1, maxIterations);

    % Initialize the start time for performance evaluation
    startTime = tic;

    %% Step 2: NSGA-II Optimization Loop
    for iter = 1:maxIterations
        prevBestFitness = Inf;

        % Step 2.1: Evaluate Fitness for Each Individual
        population = evaluatePopulationFitness(population, numTasks, numVehicles, ...
                                               computingCapacity, taskWorkload, taskDeadline);

        % Step 2.2: Perform Non-Dominated Sorting
        [fronts, rank] = nonDominatedSorting(population);

        % Step 2.3: Crowding Distance Assignment
        population = crowdingDistanceAssignment(fronts, population);

        % Step 2.4: Selection, Crossover, and Mutation
        parentPopulation = tournamentSelection(population, popSize);
        offspringPopulation = crossoverAndMutation(parentPopulation, numVehicles, numTasks);

        % Step 2.5: Combine Parent and Offspring Populations
        combinedPopulation = [population; offspringPopulation];

        % Step 2.6: Perform Non-Dominated Sorting on Combined Population
        [combinedFronts, combinedRank] = nonDominatedSorting(combinedPopulation);

        % Step 2.7: Crowding Distance for Combined Population
        combinedPopulation = crowdingDistanceAssignment(combinedFronts, combinedPopulation);

        % Step 2.8: Select Next Generation
        population = selectNextGeneration(combinedPopulation, combinedFronts, popSize);

        % Step 2.9: Update Metrics
        %bestFitness = min([population{1}.fitness(1) + population{1}.fitness(2)]);  % Pareto optimal
        bestFitness = min([population(1).fitness(1) + population(1).fitness(2)]);  % Pareto optimal

        %totalDelay = totalDelay + population{1}.fitness(1);
        totalDelay = totalDelay + population(1).fitness(1);

        totalEnergy = totalEnergy + population(1).fitness(2);

        delays(iter) = population(1).fitness(1);
        energies(iter) = population(1).fitness(2);

        % Display progress
        fprintf('Iteration %d: Best Fitness (Delay + Energy): %.2f\n', iter, bestFitness);

        % Convergence check
        if abs(prevBestFitness - bestFitness) < 1e-3
            fprintf('Convergence reached at iteration %d\n', iter);
            break;
        end
        prevBestFitness = bestFitness;
    end

    %% Step 3: Output Results
    executionTime = toc(startTime);
    averageDelay = totalDelay / maxIterations;
    averageEnergy = totalEnergy / maxIterations;
    stdDevDelay = std(delays);
    stdDevEnergy = std(energies);

    % Display final results
    fprintf('Best Solution: Delay: %.2f, Energy: %.2f\n', population(1).fitness(1), population(1).fitness(2));
    fprintf('Average Delay: %.2f, Average Energy: %.2f\n', averageDelay, averageEnergy);
    fprintf('Standard Deviation (Delay): %.2f, (Energy): %.2f\n', stdDevDelay, stdDevEnergy);
    fprintf('Execution Time: %.2f seconds\n', executionTime);

    % Return results
    result = struct('delay', averageDelay, 'energy', averageEnergy);
end

%% Helper Functions
function population = initializePopulation(popSize, numVehicles, numTasks)
    population = struct();
    for i = 1:popSize
        population(i).position = randi([1, numVehicles], 1, numTasks);
        population(i).fitness = [0, 0];
        population(i).crowdingDistance = 0;
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

function [fronts, rank] = nonDominatedSorting(population)
    % Non-dominated sorting logic
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

    fronts = front;
    rank = zeros(1, numIndividuals);
    for i = 1:numIndividuals
        rank(i) = length(front{i});
    end
end

function result = dominates(ind1, ind2)
    result = all(ind1.fitness <= ind2.fitness) && any(ind1.fitness < ind2.fitness);
end

function population = crowdingDistanceAssignment(fronts, population)
    % Compute crowding distances for each front
    for i = 1:length(fronts)
        front = fronts{i};
        if length(front) > 2
            % Calculate crowding distance for each individual in the front
            for j = 1:length(front)
                population(front(j)).crowdingDistance = rand(); % Placeholder for actual logic
            end
        end
    end
end

function selectedParents = tournamentSelection(population, numParents)
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

function nextGeneration = selectNextGeneration(population, fronts, popSize)
    % Select the next generation based on non-domination and crowding distance
    nextGeneration = population(1:popSize); % Placeholder logic for selection
end
