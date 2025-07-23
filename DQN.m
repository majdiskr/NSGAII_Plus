function [delay, energy] = DQN(numVehicles, numTasks, maxIterations, ...
                              w_delay, w_energy, taskWorkload, taskDeadline, ...
                              taskStorage, storageCapacity, computingCapacity)
    % DQN Implementation with memory-efficient settings
    % Parameters
    hiddenLayerSize = 32;          % Reduced network size
    batchSize = 32;
    memoryCapacity = 1000;
    gamma = 0.9;
    epsilon = 1.0;
    epsilon_min = 0.01;
    epsilon_decay = 0.995;
    updateTargetEvery = 10;
    
    % State dimensions
    stateSize = numTasks*2 + numVehicles;
    
    % Initialize experience replay memory
    memory = struct();
    memory.states = zeros(stateSize, memoryCapacity, 'single'); % Use single precision
    memory.actions = zeros(1, memoryCapacity, 'uint8'); % Use uint8 for actions
    memory.rewards = zeros(1, memoryCapacity, 'single');
    memory.next_states = zeros(stateSize, memoryCapacity, 'single');
    memory.dones = false(1, memoryCapacity);
    memory.count = 0;
    memory.current = 0;
    
    % Create neural network with more efficient settings
    net = feedforwardnet(hiddenLayerSize, 'trainscg'); % Scaled conjugate gradient
    net = configure(net, rand(stateSize, 1, 'single'), rand(numVehicles, 1, 'single'));
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.trainParam.epochs = 1;
    net.trainParam.max_fail = 5;
    
    targetNet = net;
    
    % Initialize metrics
    delays = zeros(1, maxIterations, 'single');
    energies = zeros(1, maxIterations, 'single');
    startTime = tic;
    
    for iter = 1:maxIterations
        % Initialize episode
        state = single(initializeState(numTasks, numVehicles));
        episodeDelay = 0;
        episodeEnergy = 0;
        
        for task = 1:numTasks
            % Epsilon-greedy action selection
            if rand() < epsilon
                action = randi([1, numVehicles]);
            else
                qValues = net(state);
                [~, action] = max(qValues);
            end
            
            % Execute action and get reward
            [delay, energy] = evaluateFitness(action, task, computingCapacity, ...
                                             taskWorkload, taskDeadline, numTasks, numVehicles);
            reward = -(w_delay*delay + w_energy*energy);
            
            % Get next state
            next_state = single(updateState(double(state), task, action, numTasks, numVehicles));
            done = (task == numTasks);
            
            % Store experience in memory
            memory.current = mod(memory.current, memoryCapacity) + 1;
            memory.states(:, memory.current) = state;
            memory.actions(memory.current) = action;
            memory.rewards(memory.current) = single(reward);
            memory.next_states(:, memory.current) = next_state;
            memory.dones(memory.current) = done;
            memory.count = min(memory.count + 1, memoryCapacity);
            
            % Train in smaller batches if memory is limited
            if memory.count >= batchSize
                batchIndices = randperm(memory.count, min(batchSize, 32)); % Smaller batch if needed
                batchStates = memory.states(:, batchIndices);
                batchActions = memory.actions(batchIndices);
                batchRewards = memory.rewards(batchIndices);
                batchNextStates = memory.next_states(:, batchIndices);
                batchDones = memory.dones(batchIndices);
                
                % Process targets in chunks
                targets = zeros(numVehicles, length(batchIndices), 'single');
                chunkSize = 8; % Process 8 samples at a time
                for chunkStart = 1:chunkSize:length(batchIndices)
                    chunkEnd = min(chunkStart+chunkSize-1, length(batchIndices));
                    chunk = chunkStart:chunkEnd;
                    
                    % Calculate target Q-values for this chunk
                    for b = chunk
                        if batchDones(b)
                            targets(batchActions(b), b) = batchRewards(b);
                        else
                            nextQ = targetNet(batchNextStates(:, b));
                            targets(batchActions(b), b) = batchRewards(b) + gamma * max(nextQ);
                        end
                    end
                    
                    % Train network on this chunk
                    net = train(net, batchStates(:, chunk), targets(:, chunk));
                end
            end
            
            % Update state and metrics
            state = next_state;
            episodeDelay = episodeDelay + delay;
            episodeEnergy = episodeEnergy + energy;
        end
        
        % Update target network
        if mod(iter, updateTargetEvery) == 0
            targetNet = net;
        end
        
        % Store metrics
        delays(iter) = episodeDelay;
        energies(iter) = episodeEnergy;
        
        % Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay);
        
        fprintf('Iteration %d: Delay %.2f, Energy %.2f, Epsilon %.2f\n', ...
                iter, episodeDelay, episodeEnergy, epsilon);
        
        % Check memory usage periodically
        if mod(iter, 10) == 0
            mem = memory;
            fprintf('Memory used: %.2f MB\n', mem.MemUsedMATLAB/1e6);
            if mem.MemUsedMATLAB > 0.9 * mem.MaxPossibleArrayBytes
                warning('Approaching memory limits, reducing batch size');
                batchSize = max(16, floor(batchSize * 0.8));
            end
        end
    end
    
    % Return average results
    delay = mean(delays);
    energy = mean(energies);
end

%% Helper Functions (same as before)
function state = initializeState(numTasks, numVehicles)
    state = zeros(numTasks*2 + numVehicles, 1);
end

function next_state = updateState(state, task, action, numTasks, numVehicles)
    next_state = state;
    next_state(task) = 1;
    next_state(numTasks + task) = action;
    next_state(2*numTasks + action) = next_state(2*numTasks + action) + 1;
end

