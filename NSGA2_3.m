function [delayNSGA2_3, energyNSGA2_3] = NSGA2_3(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                          w_delay, w_energy, taskWorkload, taskDeadline, ...
                                          taskStorage, storageCapacity, computingCapacity)

    %% Step 1: Initialize Population
    population = initializePopulation(initial_pop_size, numVehicles, numTasks);
    alpha = struct('position', [], 'fitness', [Inf, Inf]);
    totalDelay = 0;  % Accumulator for delay
    totalEnergy = 0; % Accumulator for energy
    delays = zeros(1, maxIterations);
    energies = zeros(1, maxIterations);
    
    % Convergence parameters
    bestFitness = Inf;
    convergenceThreshold = 1e-3;
    k = 10;
    minIters = 30;
    convergenceHistory = zeros(1, k);

    startTime = tic; % Start timer

    %% Step 2: NSGA-II Optimization Loop
    for iter = 1:maxIterations
        prevBestFitness = bestFitness;

        % Evaluate Fitness of Current Population
        for i = 1:length(population)
            vehicleAssignment = max(min(population(i).position, numVehicles), 1);
            [totalDelayInd, totalEnergyInd] = evaluateIndividualFitness(vehicleAssignment, ...
                                                                        numTasks, numVehicles, ...
                                                                        computingCapacity, ...
                                                                        taskWorkload, taskDeadline);
            population(i).fitness = [totalDelayInd, totalEnergyInd];
        end

        % Perform Non-Dominated Sorting
        sortedPopulation = nonDominatedSorting(population);
        sortedPopulation = calculateCrowdingDistance(sortedPopulation);

        % Update alpha solution
        if sortedPopulation(1).fitness(1) < alpha.fitness(1) && sortedPopulation(1).fitness(2) <= alpha.fitness(2)
            alpha = sortedPopulation(1);
        end

        % Combine populations for next generation
        parentPopulation = tournamentSelection(population, initial_pop_size);
        offspringPopulation = crossoverAndMutation(parentPopulation, numVehicles, numTasks);
        combinedPopulation = [population; offspringPopulation];
        sortedCombinedPopulation = nonDominatedSorting(combinedPopulation);
        sortedCombinedPopulation = calculateCrowdingDistance(sortedCombinedPopulation);
        population = selectNextGeneration(sortedCombinedPopulation, initial_pop_size);

        % Update best fitness and track statistics
        bestFitness = alpha.fitness(1) + alpha.fitness(2);
        totalDelay = totalDelay + alpha.fitness(1);
        totalEnergy = totalEnergy + alpha.fitness(2);
        delays(iter) = alpha.fitness(1);
        energies(iter) = alpha.fitness(2);

        fprintf('Iteration %d: Best Fitness (Delay + Energy) NSGA2_3: %.4f\n', iter, bestFitness);

        % Update convergence history
        fitnessChange = abs(prevBestFitness - bestFitness);
        convergenceHistory(mod(iter - 1, k) + 1) = fitnessChange;

        if iter >= max(k, minIters) && all(convergenceHistory < convergenceThreshold)
            fprintf('Convergence reached (\u0394fitness \u2264 %.4f for %d iterations) at iteration NSGA2_3 %d\n', ...
                    convergenceThreshold, k, iter);
            break;
        end
    end

    %% Output Results
    executionTime = toc(startTime);
    averageEnergy = totalEnergy / iter;
    averageDelay = totalDelay / iter;
    stdDevEnergy = std(energies(1:iter));
    stdDevDelay = std(delays(1:iter));

    fprintf('Best solution found with Delay_NSGA2_3: %.2f and Energy: %.2f\n', alpha.fitness(1), alpha.fitness(2));
    fprintf('Average Energy Consumption_NSGA2_3: %.2f\n', averageEnergy);
    fprintf('Average Delay_NSGA2_3: %.2f\n', averageDelay);
    fprintf('Standard Deviation of Energy NSGA2_3: %.2f\n', stdDevEnergy);
    fprintf('Standard Deviation of Delay NSGA2_3: %.2f\n', stdDevDelay);
    fprintf('Total Execution Time NSGA2_3: %.2f seconds\n', executionTime);

    delayNSGA2_3 = averageDelay;
    energyNSGA2_3 = averageEnergy;
end

%% Helper Functions
function population = initializePopulation(popSize, numVehicles, numTasks)
    for i = 1:popSize
        population(i).position = randi([1, numVehicles], 1, numTasks);
        population(i).fitness = [0, 0];
        population(i).crowdingDistance = 0;
    end
end

function [totalDelay, totalEnergy] = evaluateIndividualFitness(vehicleAssignment, numTasks, numVehicles, computingCapacity, taskWorkload, taskDeadline)
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
    for i = 1:numParents
        idx = randperm(length(population), 2);
        if dominates(population(idx(1)), population(idx(2)))
            selectedParents(i) = population(idx(1));
        else
            selectedParents(i) = population(idx(2));
        end
    end
end

function offspringPopulation = crossoverAndMutation(parents, numVehicles, numTasks)
    offspringPopulation = parents;
    for i = 1:length(parents)
        p1 = parents(randi(length(parents)));
        p2 = parents(randi(length(parents)));
        cp = randi(numTasks - 1);
        offspringPopulation(i).position = [p1.position(1:cp), p2.position(cp+1:end)];
        mutationRate = 0.1;
        for j = 1:numTasks
            if rand < mutationRate
                offspringPopulation(i).position(j) = randi(numVehicles);
            end
        end
    end
end

function sortedPopulation = nonDominatedSorting(population)
    N = length(population);
    front = cell(1, N);
    domCount = zeros(1, N);
    dominated = cell(1, N);

    for p = 1:N
        dominated{p} = [];
        domCount(p) = 0;
        for q = 1:N
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
    sortedPopulation = population([front{:}]);
end

function population = calculateCrowdingDistance(population)
    N = length(population);
    if N < 3
        for i = 1:N
            population(i).crowdingDistance = Inf;
        end
        return;
    end

    for m = 1:2
        f = [population.fitness];
        f = reshape(f, 2, []).';
        [~, idx] = sort(f(:, m));
        population(idx(1)).crowdingDistance = Inf;
        population(idx(end)).crowdingDistance = Inf;
        fmax = f(idx(end), m);
        fmin = f(idx(1), m);
        if fmax - fmin == 0
            continue;
        end
        for i = 2:N-1
            population(idx(i)).crowdingDistance = population(idx(i)).crowdingDistance + ...
                (f(idx(i+1), m) - f(idx(i-1), m)) / (fmax - fmin);
        end
    end
end

function nextGen = selectNextGeneration(population, popSize)
    [~, idx] = sort([population.crowdingDistance], 'descend');
    nextGen = population(idx(1:popSize));
end

function result = dominates(ind1, ind2)
    result = all(ind1.fitness <= ind2.fitness) && any(ind1.fitness < ind2.fitness);
end
