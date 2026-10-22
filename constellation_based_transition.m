%% CONSTELLATION-DISTANCE-BASED MARKOV MODEL FOR GF(8) TWO-VN STATE SPACE
% State:
%   X_t = (x26_hat, x31_hat)
%
% Each VN takes values in GF(8) = {0,1,2,3,4,5,6,7}
% Total states = 8^2 = 64
%
% Transition rule:
%   At each iteration, choose one VN randomly.
%   Replace that VN by another GF(8) symbol with probability based on
%   the Euclidean distance between 8-QAM constellation points.
%
% This model does NOT use syndrome.
% It only uses constellation closeness.

clear; clc; close all;

%% ------------------------------------------------------------------------
% Build all 64 states
%% ------------------------------------------------------------------------

states = zeros(64,2);

idx = 1;
for a = 0:7
    for b = 0:7
        states(idx,:) = [a,b];
        idx = idx + 1;
    end
end

numStates = size(states,1);

%% ------------------------------------------------------------------------
% Define correct and initial states
%% ------------------------------------------------------------------------

correctSymbolPair = [2, 7];

% Dominant observed wrong transition:
% true:  (2,7)
% wrong: (1,3)
initialState = [1, 3];

correctState = find_state_index(states, correctSymbolPair);
initialIndex = find_state_index(states, initialState);

fprintf('Total number of states = %d\n', numStates);
fprintf('Correct state = (%d,%d), index = %d\n', ...
    correctSymbolPair(1), correctSymbolPair(2), correctState);
fprintf('Initial state = (%d,%d), index = %d\n', ...
    initialState(1), initialState(2), initialIndex);

%% ------------------------------------------------------------------------
% Define 8-QAM constellation
% Here simple rectangular 8-QAM:
%
%   0 -> -3 + 1j
%   1 -> -1 + 1j
%   2 ->  1 + 1j
%   3 ->  3 + 1j
%   4 -> -3 - 1j
%   5 -> -1 - 1j
%   6 ->  1 - 1j
%   7 ->  3 - 1j
%
% Then normalize to unit average energy.
%% ------------------------------------------------------------------------

constellation = [ ...
    -3 + 1j; ...
    -1 + 1j; ...
     1 + 1j; ...
     3 + 1j; ...
    -3 - 1j; ...
    -1 - 1j; ...
     1 - 1j; ...
     3 - 1j ];

% Normalize average symbol energy to 1
Es_avg = mean(abs(constellation).^2);
constellation = constellation / sqrt(Es_avg);

%% ------------------------------------------------------------------------
% Build symbol transition probability matrix based on constellation distance
%
% Psym(a+1,b+1) = probability of changing symbol a to symbol b.
%
% tau controls how strongly nearby constellation points are preferred.
% Smaller tau = stronger nearest-neighbor preference.
% Larger tau = more uniform/random.
%% ------------------------------------------------------------------------

tau = 0.30;

Psym = zeros(8,8);

for a = 0:7

    d2 = zeros(1,8);

    for b = 0:7
        d2(b+1) = abs(constellation(a+1) - constellation(b+1))^2;
    end

    weights = exp(-d2 / tau);

    % Optional: prevent staying in same symbol during an update
    % If you want the chosen VN to always change, uncomment this:
    % weights(a+1) = 0;

    Psym(a+1,:) = weights / sum(weights);
end

disp('Symbol transition probability matrix Psym:');
disp(Psym);

%% ------------------------------------------------------------------------
% Build 64 x 64 state transition matrix
%
% At each step:
%   choose v26 or v31 with probability 1/2
%   update the selected symbol according to Psym
%% ------------------------------------------------------------------------

P = zeros(numStates,numStates);

for i = 1:numStates

    x26 = states(i,1);
    x31 = states(i,2);

    % Update v26
    for newSym = 0:7
        newState = [newSym, x31];
        j = find_state_index(states, newState);

        P(i,j) = P(i,j) + (1/2) * Psym(x26+1,newSym+1);
    end

    % Update v31
    for newSym = 0:7
        newState = [x26, newSym];
        j = find_state_index(states, newState);

        P(i,j) = P(i,j) + (1/2) * Psym(x31+1,newSym+1);
    end
end

%% ------------------------------------------------------------------------
% Check transition matrix
%% ------------------------------------------------------------------------

rowSums = sum(P,2);
fprintf('\nMaximum row-sum error = %.3e\n', max(abs(rowSums - 1)));

%% ------------------------------------------------------------------------
% Probability of being in correct state after T iterations
%% ------------------------------------------------------------------------

T = 30;
Tmax = 100;

p0 = zeros(1,numStates);
p0(initialIndex) = 1;

pT = p0 * (P^T);

fprintf('\nInitial state = (%d,%d)\n', initialState(1), initialState(2));
fprintf('Correct state = (%d,%d)\n', correctSymbolPair(1), correctSymbolPair(2));
fprintf('tau = %.3f\n', tau);
fprintf('Probability of being in correct state after T=%d iterations = %.6f\n', ...
    T, pT(correctState));

%% ------------------------------------------------------------------------
% Probability history over iterations
%% ------------------------------------------------------------------------

p = p0;
probCorrectHistory = zeros(Tmax,1);

for t = 1:Tmax
    p = p * P;
    probCorrectHistory(t) = p(correctState);
end

fprintf('\nFinite-time probabilities:\n');
for Tcheck = [1 2 3 5 10 20 23 30 50 100]
    fprintf('T = %3d, P_correct = %.6f\n', ...
        Tcheck, probCorrectHistory(Tcheck));
end

figure;
plot(1:Tmax, probCorrectHistory, 'o-', 'LineWidth', 1.5);
xlabel('Iteration number T');
ylabel(sprintf('P{X_T = (%d,%d)}', correctSymbolPair(1), correctSymbolPair(2)));
title(sprintf('8-QAM-distance Markov model from X_0 = (%d,%d)', ...
    initialState(1), initialState(2)));
grid on;

%% ------------------------------------------------------------------------
% Probability from every initial state after T
%% ------------------------------------------------------------------------

PT = P^T;
probCorrectFromEachState = PT(:,correctState);

resultTable = table(states(:,1), states(:,2), probCorrectFromEachState, ...
    'VariableNames', {'x26_hat_initial','x31_hat_initial','P_correct_after_T'});

disp(resultTable);

writetable(resultTable, 'qam_distance_GF8_4cycle_probabilities.csv');

%% ------------------------------------------------------------------------
% Plot constellation with labels
%% ------------------------------------------------------------------------

figure;
plot(real(constellation), imag(constellation), 'o', 'MarkerSize', 8, 'LineWidth', 1.5);
hold on;

for a = 0:7
    text(real(constellation(a+1))+0.03, imag(constellation(a+1))+0.03, ...
        sprintf('%d', a), 'FontSize', 12);
end

xlabel('In-phase');
ylabel('Quadrature');
title('Normalized 8-QAM constellation labeling');
grid on;
axis equal;

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


%%%%%%%%%%% mapping %%%%% check with actual decoder mapping

%0 -> -3 + 1j
%1 -> -1 + 1j
%2 ->  1 + 1j
%3 ->  3 + 1j
%4 -> -3 - 1j
%5 -> -1 - 1j
%6 ->  1 - 1j
%7 ->  3 - 1j