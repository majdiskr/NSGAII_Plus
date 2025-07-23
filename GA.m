function [delay, energy] = GA(numVehicles, numTasks, populationSize, maxIterations, ...
                             w_delay, w_energy, taskWorkload, taskDeadline, ...
                             taskStorage, storageCapacity, computingCapacity)
    % GA parameters
    crossoverProb = 0.8;
    mutationProb = 0.1;
    tournamentSize = 3;
    
    % Initialize population
    population = struct('chromosome', {}, 'fitness', {});
    for i = 1:populationSize
        population(i).chromosome = randi([1, numVehicles], 1, numTasks);
        population(i).fitness = 0;
    end
    
    % Initialize metrics
    delays = zeros(1, maxIterations);
    energies = zeros(1, maxIterations);
    totalDelay = 0;
    totalEnergy = 0;
    startTime = tic;
    
    for iter = 1:maxIterations
        % Evaluate fitness
        for i = 1:populationSize
            [delay, energy] = evaluateChromosome(population(i).chromosome, numTasks, computingCapacity, ...
                                               taskWorkload, taskDeadline, numVehicles);
            population(i).fitness = w_delay*delay + w_energy*energy;
        end
        
        % Sort population by fitness
        [~, idx] = sort([population.fitness]);
        population = population(idx);
        
        % Record best solution
        bestChromosome = population(1).chromosome;
        [bestDelay, bestEnergy] = evaluateChromosome(bestChromosome, numTasks, computingCapacity, ...
                                                    taskWorkload, taskDeadline, numVehicles);
        delays(iter) = bestDelay;
        energies(iter) = bestEnergy;
        totalDelay = totalDelay + bestDelay;
        totalEnergy = totalEnergy + bestEnergy;
        
        % Create new population with proper initialization
        newPopulation = struct('chromosome', cell(1, populationSize), 'fitness', cell(1, populationSize));
        newPopulation(1).chromosome = population(1).chromosome; % Elitism
        newPopulation(1).fitness = population(1).fitness;
        
        for i = 2:populationSize
            % Selection (Tournament selection)
            parent1 = tournamentSelection(population, tournamentSize);
            parent2 = tournamentSelection(population, tournamentSize);
            
            % Crossover
            if rand() < crossoverProb
                [child1, child2] = crossover(parent1.chromosome, parent2.chromosome);
            else
                child1 = parent1.chromosome;
                child2 = parent2.chromosome;
            end
            
            % Mutation
            if rand() < mutationProb
                child1 = mutate(child1, numVehicles);
            end
            if rand() < mutationProb
                child2 = mutate(child2, numVehicles);
            end
            
            % Select one child randomly
            if rand() < 0.5
                newPopulation(i).chromosome = child1;
            else
                newPopulation(i).chromosome = child2;
            end
            
            % Evaluate new chromosome's fitness
            [delay, energy] = evaluateChromosome(newPopulation(i).chromosome, numTasks, computingCapacity, ...
                                               taskWorkload, taskDeadline, numVehicles);
            newPopulation(i).fitness = w_delay*delay + w_energy*energy;
        end
        
        population = newPopulation;
        
        fprintf('Iteration %d: Best Delay %.2f, Energy %.2f\n', iter, bestDelay, bestEnergy);
    end
    
    % Calculate final metrics
    executionTime = toc(startTime);
    averageEnergy = mean(energies);
    averageDelay = mean(delays);
    stdDevEnergy = std(energies);
    stdDevDelay = std(delays);
    
    % Output Results
    fprintf('\nGA Final Results:\n');
    fprintf('Average Delay: %.2f ± %.2f\n', averageDelay, stdDevDelay);
    fprintf('Average Energy: %.2f ± %.2f\n', averageEnergy, stdDevEnergy);
    fprintf('Execution Time: %.2f seconds\n', executionTime);
    
    delay = averageDelay;
    energy = averageEnergy;
end

%% Helper Functions
function [delay, energy] = evaluateChromosome(chromosome, numTasks, computingCapacity, ...
                                            taskWorkload, taskDeadline, numVehicles)
    delay = 0;
    energy = 0;
    for task = 1:numTasks
        [d, e] = evaluateFitness(chromosome(task), task, computingCapacity, ...
                               taskWorkload, taskDeadline, numTasks, numVehicles);
        delay = delay + d;
        energy = energy + e;
    end
end

function selected = tournamentSelection(population, tournamentSize)
    % Tournament selection
    contenders = randperm(length(population), tournamentSize);
    [~, idx] = min([population(contenders).fitness]);
    selected = population(contenders(idx));
end

function [child1, child2] = crossover(parent1, parent2)
    % Single-point crossover
    point = randi([1, length(parent1)-1]);
    child1 = [parent1(1:point), parent2(point+1:end)];
    child2 = [parent2(1:point), parent1(point+1:end)];
end

function mutated = mutate(chromosome, numVehicles)
    % Random mutation
    point = randi([1, length(chromosome)]);
    mutated = chromosome;
    mutated(point) = randi([1, numVehicles]);
end