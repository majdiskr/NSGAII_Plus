function [delay, energy] = PSO(numVehicles, numTasks, swarmSize, maxIterations, ...
                              w_delay, w_energy, taskWorkload, taskDeadline, ...
                              taskStorage, storageCapacity, computingCapacity)
    % PSO parameters
    w = 0.7; % Inertia weight
    c1 = 1.5; % Cognitive coefficient
    c2 = 1.5; % Social coefficient
    
    % Initialize swarm
    swarm = struct();
    for i = 1:swarmSize
        swarm(i).position = randi([1, numVehicles], 1, numTasks);
        swarm(i).velocity = zeros(1, numTasks);
        swarm(i).bestPosition = swarm(i).position;
        [delay, energy] = evaluatePosition(swarm(i).position, numTasks, computingCapacity, ...
                                         taskWorkload, taskDeadline, numVehicles);
        swarm(i).bestFitness = w_delay*delay + w_energy*energy;
    end
    
    % Initialize global best
    [~, idx] = min([swarm.bestFitness]);
    globalBestPosition = swarm(idx).bestPosition;
    globalBestFitness = swarm(idx).bestFitness;
    
    % Initialize metrics
    delays = zeros(1, maxIterations);
    energies = zeros(1, maxIterations);
    totalDelay = 0;
    totalEnergy = 0;
    startTime = tic;
    
    for iter = 1:maxIterations
        for i = 1:swarmSize
            % Update velocity
            r1 = rand(1, numTasks);
            r2 = rand(1, numTasks);
            cognitive = c1 * r1 .* (swarm(i).bestPosition - swarm(i).position);
            social = c2 * r2 .* (globalBestPosition - swarm(i).position);
            swarm(i).velocity = w * swarm(i).velocity + cognitive + social;
            
            % Update position (discrete PSO)
            newPosition = round(swarm(i).position + swarm(i).velocity);
            newPosition = max(min(newPosition, numVehicles), 1); % Clamp to valid range
            
            % Evaluate new position
            [delay, energy] = evaluatePosition(newPosition, numTasks, computingCapacity, ...
                                             taskWorkload, taskDeadline, numVehicles);
            newFitness = w_delay*delay + w_energy*energy;
            
            % Update personal best
            if newFitness < swarm(i).bestFitness
                swarm(i).position = newPosition;
                swarm(i).bestPosition = newPosition;
                swarm(i).bestFitness = newFitness;
                
                % Update global best
                if newFitness < globalBestFitness
                    globalBestPosition = newPosition;
                    globalBestFitness = newFitness;
                end
            else
                swarm(i).position = newPosition;
            end
        end
        
        % Record best solution
        [bestDelay, bestEnergy] = evaluatePosition(globalBestPosition, numTasks, computingCapacity, ...
                                                 taskWorkload, taskDeadline, numVehicles);
        delays(iter) = bestDelay;
        energies(iter) = bestEnergy;
        totalDelay = totalDelay + bestDelay;
        totalEnergy = totalEnergy + bestEnergy;
        
        fprintf('Iteration %d: Best Delay %.2f, Energy %.2f\n', iter, bestDelay, bestEnergy);
    end
    
    % Calculate final metrics
    executionTime = toc(startTime);
    averageEnergy = totalEnergy / maxIterations;
    averageDelay = totalDelay / maxIterations;
    stdDevEnergy = std(energies);
    stdDevDelay = std(delays);
    
    % Output Results
    fprintf('PSO Results:\n');
    fprintf('Average Delay: %.2f\n', averageDelay);
    fprintf('Average Energy: %.2f\n', averageEnergy);
    fprintf('Standard Deviation of Energy: %.2f\n', stdDevEnergy);
    fprintf('Standard Deviation of Delay: %.2f\n', stdDevDelay);
    fprintf('Total Execution Time: %.2f seconds\n', executionTime);
    
    delay = averageDelay;
    energy = averageEnergy;
end

function [delay, energy] = evaluatePosition(position, numTasks, computingCapacity, ...
                                         taskWorkload, taskDeadline, numVehicles)
    delay = 0;
    energy = 0;
    for task = 1:numTasks
        [d, e] = evaluateFitness(position(task), task, computingCapacity, ...
                               taskWorkload, taskDeadline, numTasks, numVehicles);
        delay = delay + d;
        energy = energy + e;
    end
end
function [delay, energy] = evaluateFitness(vehicle, task, computingCapacity, taskWorkload, taskDeadline, numTasks, numVehicles)
    % Precompute values that remain constant for all tasks/vehicles
    workload = taskWorkload(task);  % Task workload for the specific task
    
    % Efficient delay calculation
    delay = workload / computingCapacity(vehicle);  % Delay is directly proportional to workload and inversely to computing capacity
    
    % Efficient energy calculation
    scalingFactor = (taskDeadline(task) / 10) * (1 + task / numTasks);
    energy = workload * scalingFactor;  % Energy is based on workload and a scaling factor derived from deadline and task index
end
