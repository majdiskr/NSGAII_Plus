function [delayMOEAD, energyMOEAD] = MOEAD(numVehicles, numTasks, initial_pop_size, someFactor, maxIterations, ...
                                           w_delay, w_energy, taskWorkload, taskDeadline, ...
                                           taskStorage, storageCapacity, computingCapacity)
    %% Step 1: Initialize Population and Weight Vectors
    population = initializePopulation(initial_pop_size, numVehicles, numTasks);
    weights = [linspace(0, 1, initial_pop_size); linspace(1, 0, initial_pop_size)]'; % Shape: initial_pop_size x 2
    totalDelay = 0;
    totalEnergy = 0;
    delays = zeros(1, maxIterations);
    energies = zeros(1, maxIterations);
    bestFitness = Inf;
    convergenceThreshold = 1e-3;
    startTime = tic;

    %% Step 2: MOEA/D Optimization Loop
    for iter = 1:maxIterations
        prevBestFitness = bestFitness;

        % Evaluate Fitness of Current Population
        popFitness = zeros(initial_pop_size, 2); % Preallocate fitness matrix
        for i = 1:length(population)
            vehicleAssignment = max(min(round(population(i).position), numVehicles), 1);
            [delayInd, energyInd] = evaluateIndividualFitness(vehicleAssignment, ...
                                                              numTasks, numVehicles, ...
                                                              computingCapacity, ...
                                                              taskWorkload, taskDeadline);
            population(i).fitness = [delayInd, energyInd];
            popFitness(i, :) = [delayInd, energyInd];
            z = min(popFitness, [], 1); % Initial ideal point
            if length(z) < 2
                z = [0, 0]; % Fallback
            end
            population(i).scalarFitness = max(weights(i,1) * abs(delayInd - z(1)), ...
                                             weights(i,2) * abs(energyInd - z(2)));
        end

        % Update Neighboring Solutions
        T = 5; % Neighborhood size
        for i = 1:length(population)
            neighbors = randperm(initial_pop_size, min(T, initial_pop_size));
            for n = neighbors
                if population(n).scalarFitness < population(i).scalarFitness
                    population(i).position = population(n).position;
                    population(i).fitness = population(n).fitness;
                    popFitness(i, :) = population(n).fitness;
                    population(i).scalarFitness = population(n).scalarFitness;
                end
            end
        end

        % Generate and Evaluate Offspring
        offspringPopulation = crossoverAndMutation(population, numVehicles, numTasks);
        offspringFitness = zeros(length(offspringPopulation), 2); % Preallocate offspring fitness
        for i = 1:length(offspringPopulation)
            vehicleAssignment = max(min(round(offspringPopulation(i).position), numVehicles), 1);
            [delayInd, energyInd] = evaluateIndividualFitness(vehicleAssignment, ...
                                                              numTasks, numVehicles, ...
                                                              computingCapacity, ...
                                                              taskWorkload, taskDeadline);
            offspringPopulation(i).fitness = [delayInd, energyInd];
            offspringFitness(i, :) = [delayInd, energyInd];
            % Update z using evaluated solutions
            combinedFitness = [popFitness; offspringFitness(1:i, :)];
            z = min(combinedFitness, [], 1);
            if length(z) < 2
                z = [0, 0]; % Fallback
            end
            offspringPopulation(i).scalarFitness = max(weights(mod(i-1, initial_pop_size)+1,1) * abs(delayInd - z(1)), ...
                                                      weights(mod(i-1, initial_pop_size)+1,2) * abs(energyInd - z(2)));
        end

        % Update Population
        combinedPopulation = [population; offspringPopulation];
        combinedFitness = [popFitness; offspringFitness];
        sortedPopulation = sortByScalarFitness(combinedPopulation);
        population = sortedPopulation(1:initial_pop_size);
        popFitness = combinedFitness(1:initial_pop_size, :); % Update popFitness

        % Update Best Fitness and Accumulate Results
        bestFitness = min([population.scalarFitness]);
        totalDelay = totalDelay + population(1).fitness(1);
        totalEnergy = totalEnergy + population(1).fitness(2);
        delays(iter) = population(1).fitness(1);
        energies(iter) = population(1).fitness(2);

        % Display Progress
        fprintf('Iteration %d: Best Fitness (Scalar) MOEAD: %.2f\n', iter, bestFitness);

        % Check for Convergence
        if abs(prevBestFitness - bestFitness) < convergenceThreshold
            fprintf('Convergence reached at iteration MOEAD %d\n', iter);
            break;
        end
    end

    %% Output Results
    executionTime = toc(startTime);
    averageDelay = totalDelay / maxIterations;
    averageEnergy = totalEnergy / maxIterations;
    stdDevDelay = std(delays);
    stdDevEnergy = std(energies);

    fprintf('Best solution found with Delay_MOEAD: %.2f and Energy: %.2f\n', population(1).fitness(1), population(1).fitness(2));
    fprintf('Average Delay_MOEAD: %.2f\n', averageDelay);
    fprintf('Average Energy Consumption_MOEAD: %.2f\n', averageEnergy);
    fprintf('Standard Deviation of Delay MOEAD: %.2f\n', stdDevDelay);
    fprintf('Standard Deviation of Energy MOEAD: %.2f\n', stdDevEnergy);
    fprintf('Total Execution Time MOEAD: %.2f seconds\n', executionTime);

    % Return delay and energy
    delayMOEAD = averageDelay;
    energyMOEAD = averageEnergy;
end

%% Helper Functions
function population = initializePopulation(popSize, numVehicles, numTasks)
    population = struct();
    for i = 1:popSize
        population(i).position = randi([1, numVehicles], 1, numTasks);
        population(i).fitness = [0, 0];
        population(i).scalarFitness = Inf;
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

function sortedPopulation = sortByScalarFitness(population)
    [~, idx] = sort([population.scalarFitness]);
    sortedPopulation = population(idx);
end