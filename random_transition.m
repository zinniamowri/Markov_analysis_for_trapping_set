%% RANDOM TRANSITION MARKOV MODEL FOR GF(8) TWO-VN STATE SPACE
% State:
%   X_t = (x26_hat, x31_hat)
%
% Each VN can take values in GF(8) = {0,1,2,3,4,5,6,7}
% Total number of states = 8^2 = 64
%
% This model does NOT use syndrome calculation.
% It is a pure random state-space baseline.

clear; clc; close all;

%% Build all 64 states
states = zeros(64,2);

idx = 1;
for a = 0:7
    for b = 0:7
        states(idx,:) = [a,b];
        idx = idx + 1;
    end
end

numStates = size(states,1);

%% Correct and initial states
correctSymbolPair = [2, 7];
initialState = [1, 3];   % dominant observed wrong state: (2->1, 7->3)

correctState = find_state_index(states, correctSymbolPair);
initialIndex = find_state_index(states, initialState);

fprintf('Total number of states = %d\n', numStates);
fprintf('Correct state = (%d,%d), index = %d\n', ...
    correctSymbolPair(1), correctSymbolPair(2), correctState);
fprintf('Initial state = (%d,%d), index = %d\n', ...
    initialState(1), initialState(2), initialIndex);

%% Parameters
T = 30;
Tmax = 100;

%% ------------------------------------------------------------------------
% Case 1: Fully random jump to any of the 64 states
%
% From every state, the next state is chosen uniformly from all 64 states.
%
% P_ij = 1/64 for all i,j
%% ------------------------------------------------------------------------

P_fullRandom = ones(numStates,numStates) / numStates;

rowSums = sum(P_fullRandom,2);
fprintf('\nFULL RANDOM JUMP MODEL\n');
fprintf('Maximum row-sum error = %.3e\n', max(abs(rowSums - 1)));

p0 = zeros(1,numStates);
p0(initialIndex) = 1;

pT = p0 * (P_fullRandom^T);

fprintf('Probability of being in correct state after T=%d iterations = %.6f\n', ...
    T, pT(correctState));

%% Probability history for full random model
p = p0;
probCorrect_fullRandom = zeros(Tmax,1);

for t = 1:Tmax
    p = p * P_fullRandom;
    probCorrect_fullRandom(t) = p(correctState);
end

%% ------------------------------------------------------------------------
% Case 2: Random one-symbol update
%
% At each step:
%   choose v26 or v31 with probability 1/2
%   replace only that chosen VN by a random GF(8) symbol
%
% This is still random, but more similar to symbol-flipping,
% because only one VN changes at a time.
%% ------------------------------------------------------------------------

P_oneSymbolRandom = zeros(numStates,numStates);

for i = 1:numStates

    x26 = states(i,1);
    x31 = states(i,2);

    for vn = [26,31]

        pChooseVN = 1/2;

        for sym = 0:7

            if vn == 26
                newState = [sym, x31];
            else
                newState = [x26, sym];
            end

            j = find_state_index(states, newState);

            P_oneSymbolRandom(i,j) = P_oneSymbolRandom(i,j) + pChooseVN * (1/8);
        end
    end
end

rowSums = sum(P_oneSymbolRandom,2);
fprintf('\nONE-SYMBOL RANDOM UPDATE MODEL\n');
fprintf('Maximum row-sum error = %.3e\n', max(abs(rowSums - 1)));

pT = p0 * (P_oneSymbolRandom^T);

fprintf('Probability of being in correct state after T=%d iterations = %.6f\n', ...
    T, pT(correctState));

%% Probability history for one-symbol random model
p = p0;
probCorrect_oneSymbolRandom = zeros(Tmax,1);

for t = 1:Tmax
    p = p * P_oneSymbolRandom;
    probCorrect_oneSymbolRandom(t) = p(correctState);
end

%% Print finite-time probabilities
fprintf('\nFinite-time probabilities: FULL RANDOM JUMP\n');
for Tcheck = [1 2 3 5 10 20 23 30 50 100]
    fprintf('T = %3d, P_correct = %.6f\n', ...
        Tcheck, probCorrect_fullRandom(Tcheck));
end

fprintf('\nFinite-time probabilities: ONE-SYMBOL RANDOM UPDATE\n');
for Tcheck = [1 2 3 5 10 20 23 30 50 100]
    fprintf('T = %3d, P_correct = %.6f\n', ...
        Tcheck, probCorrect_oneSymbolRandom(Tcheck));
end

%% Plot comparison
figure;
plot(1:Tmax, probCorrect_fullRandom, 'o-', 'LineWidth', 1.5);
hold on;
plot(1:Tmax, probCorrect_oneSymbolRandom, 's-', 'LineWidth', 1.5);
xlabel('Iteration number T');
ylabel(sprintf('P{X_T = (%d,%d)}', correctSymbolPair(1), correctSymbolPair(2)));
title(sprintf('Random Markov models from X_0 = (%d,%d)', ...
    initialState(1), initialState(2)));
legend('Full random jump', 'One-symbol random update', 'Location', 'best');
grid on;

%% Probability from every initial state after T
PT_full = P_fullRandom^T;
probCorrectFromEachState_full = PT_full(:,correctState);

PT_one = P_oneSymbolRandom^T;
probCorrectFromEachState_one = PT_one(:,correctState);

resultTable = table(states(:,1), states(:,2), ...
    probCorrectFromEachState_full, probCorrectFromEachState_one, ...
    'VariableNames', {'x26_hat_initial','x31_hat_initial', ...
    'P_correct_full_random','P_correct_one_symbol_random'});

disp(resultTable);

writetable(resultTable, 'random_GF8_4cycle_markov_probabilities.csv');

%% ------------------------------------------------------------------------
% Helper function
%% ------------------------------------------------------------------------

function idx = find_state_index(states, targetState)

    idx = find(states(:,1)==targetState(1) & ...
               states(:,2)==targetState(2), 1);

    if isempty(idx)
        error('State (%d,%d) not found.', targetState(1), targetState(2));
    end
end