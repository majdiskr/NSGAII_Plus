function [delayMOHHO, energyMOHHO] = MOHHO(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                           w_delay, w_energy, taskWorkload, taskDeadline, ...
                                           taskStorage, storageCapacity, computingCapacity)
    %% Step 1: Initialize Population (Hawks)
    population = initializePopulation(initial_pop_size, numVehicles, numTasks);
    archive = []; % Store non-dominated solutions
    totalDelay = 0;
    totalEnergy = 0;
    delays = zeros(1, maxIterations);
    energies = zeros(1, maxIterations);
    bestFitness = Inf;
    convergenceThreshold = 1e-3;
    startTime = tic;

    %% Step 2: MOHHO Optimization Loop
    for iter = 1:maxIterations
        prevBestFitness = bestFitness;

        % Evaluate Fitness of Current Population
        for i = 1:length(population)
            vehicleAssignment = max(min(round(population(i).position), numVehicles), 1); % Ensure valid discrete assignments
            [totalDelayInd, totalEnergyInd] = evaluateIndividualFitness(vehicleAssignment, ...
                                                                        numTasks, numVehicles, ...
                                                                        computingCapacity, ...
                                                                        taskWorkload, taskDeadline);
            population(i).fitness = [totalDelayInd, totalEnergyInd];
        end

        % Update Archive with Non-Dominated Solutions
        archive = [archive; population];
        archive = nonDominatedSorting(archive);
        archive = archive(1:min(length(archive), initial_pop_size)); % Limit archive size

        % Select Prey (Best Solution in Archive)
        prey = archive(1); % Simplest: take first non-dominated solution as prey

        % Update Hawk Positions
        for i = 1:length(population)
            E = 2 * (1 - iter/maxIterations); % Energy factor decreases over iterations
            r = rand(); % Random jump strength
            if E >= 1 % Exploration Phase (Random Perching)
                if r < 0.5
                    population(i).position = randi([1, numVehicles], 1, numTasks); % Random perch
                else
                    randomHawk = population(randi(initial_pop_size));
                    population(i).position = randomHawk.position - rand() * abs(randomHawk.position - population(i).position);
                end
            else % Exploitation Phase (Soft/Hard Besiege)
                delta = prey.position - population(i).position;
                if r >= 0.5 % Soft Besiege
                    population(i).position = prey.position - E * abs(delta);
                else % Hard Besiege
                    population(i).position = prey.position - E * abs(2 * rand() * prey.position - population(i).position);
                end
            end
            % Bound positions to valid vehicle indices
            population(i).position = max(min(round(population(i).position), numVehicles), 1);
        end

        % Update Best Fitness and Accumulate Results
        bestFitness = sum(archive(1).fitness); % Sum of delay and energy for simplicity
        totalDelay = totalDelay + archive(1).fitness(1);
        totalEnergy = totalEnergy + archive(1).fitness(2);
        delays(iter) = archive(1).fitness(1);
        energies(iter) = archive(1).fitness(2);

        % Display Progress
        fprintf('Iteration %d: Best Fitness (Delay + Energy) MOHHO: %.2f\n', iter, bestFitness);

        % Check for Convergence
        if abs(prevBestFitness - bestFitness) < convergenceThreshold
            fprintf('Convergence reached at iteration MOHHO %d\n', iter);
            break;
        end
    end

    %% Output Results
    executionTime = toc(startTime);
    averageDelay = totalDelay / maxIterations;
    averageEnergy = totalEnergy / maxIterations;
    stdDevDelay = std(delays);
    stdDevEnergy = std(energies);

    fprintf('Best solution found with Delay_MOHHO: %.2f and Energy: %.2f\n', archive(1).fitness(1), archive(1).fitness(2));
    fprintf('Average Delay_MOHHO: %.2f\n', averageDelay);
    fprintf('Average Energy Consumption_MOHHO: %.2f\n', averageEnergy);
    fprintf('Standard Deviation of Delay MOHHO: %.2f\n', stdDevDelay);
    fprintf('Standard Deviation of Energy MOHHO: %.2f\n', stdDevEnergy);
    fprintf('Total Execution Time MOHHO: %.2f seconds\n', executionTime);

    % Return delay and energy
    delayMOHHO = averageDelay;
    energyMOHHO = averageEnergy;
end

%% Helper Functions
function population = initializePopulation(popSize, numVehicles, numTasks)
    % Initialize the population with random assignments
    population = struct();
    for i = 1:popSize
        population(i).position = randi([1, numVehicles], 1, numTasks);
        population(i).fitness = [0, 0]; % Placeholder for [delay, energy]
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

function result = dominates(ind1, ind2)
    % Check if ind1 dominates ind2
    result = all(ind1.fitness <= ind2.fitness) && any(ind1.fitness < ind2.fitness);
end