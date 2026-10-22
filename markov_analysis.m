%% NON-ABSORBING MARKOV MODEL FOR LOCAL GF(8) 4-CYCLE
% Local 4-cycle:
% VN pair: (v26, v31)
% CN pair: (c9, c22)
%
% State:
%   X_t = (x26_hat, x31_hat)
%
% Each VN can take one of 8 symbols in GF(8).
% Therefore, total state space size = 8^2 = 64.
%
% Correct transmitted local symbol pair:
%   (x26, x31) = (2, 7)
%
% This script computes:
%   P{X_T = (2,7) | X_0 = initialState}
%
% Important:
%   The correct state is NOT absorbing in this model.

clear; clc; close all;

%% ------------------------------------------------------------------------
% GF(8) setup
% Primitive polynomial: x^3 + x + 1
% Binary representation: 1011 = decimal 11
%% ------------------------------------------------------------------------

prim_poly = 11;

gf_add = @(a,b) bitxor(a,b);
gf_mul = @(a,b) gf8_mul(a,b,prim_poly);
gf_div = @(a,b) gf8_div(a,b,prim_poly);

%% ------------------------------------------------------------------------
% Edge labels for dominant pair (v26, v31)
%% ------------------------------------------------------------------------

h926  = 3;   % h(9,26)
h931  = 5;   % h(9,31)
h2226 = 3;   % h(22,26)
h2231 = 6;   % h(22,31)

%% ------------------------------------------------------------------------
% Correct transmitted symbol pair
%% ------------------------------------------------------------------------

correctSymbolPair = [2, 7];

%% ------------------------------------------------------------------------
% Contributions from all other VNs connected to c9 and c22
% For transmitted symbols x26 = 2 and x31 = 7,

%   3(2) + 5(7) + R9  = 0  --> R9  = 0
%   3(2) + 6(7) + R22 = 0  --> R22 = 2
%
% These values are kept fixed in the local Markov model.
%% ------------------------------------------------------------------------

R9  = 0;
R22 = 2;

%% ------------------------------------------------------------------------
% Build all 64 local symbol states
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

correctState = find_state_index(states, correctSymbolPair);

fprintf('Total number of states = %d\n', numStates);
fprintf('Correct state = (%d,%d), index = %d\n', ...
    correctSymbolPair(1), correctSymbolPair(2), correctState);

%% ------------------------------------------------------------------------
% Markov model parameters
%% ------------------------------------------------------------------------

T = 500;                  % finite decoder iteration limit
epsilonRandom = 0.10;    % random/noise-driven update probability

% Interpretation:
% With probability (1 - epsilonRandom), the decoder follows CN-guided update.
% With probability epsilonRandom, it makes a random symbol update.
%
% Increase epsilonRandom to model stronger random perturbation.

%% ------------------------------------------------------------------------
% Build transition matrix P
%% ------------------------------------------------------------------------

P = zeros(numStates,numStates);

for i = 1:numStates

    x26 = states(i,1);
    x31 = states(i,2);

    % ------------------------------------------------------------
    % Local syndrome equations using actual decoded symbols
    %
    % s9  = 3*x26 + 5*x31 + R9
    % s22 = 3*x26 + 6*x31 + R22
    %
    % All operations are over GF(8).
    % ------------------------------------------------------------

    s9 = gf_add( ...
            gf_add(gf_mul(h926,x26), gf_mul(h931,x31)), ...
            R9);

    s22 = gf_add( ...
            gf_add(gf_mul(h2226,x26), gf_mul(h2231,x31)), ...
            R22);

    unsatChecks = [];

    if s9 ~= 0
        unsatChecks = [unsatChecks, 9];
    end

    if s22 ~= 0
        unsatChecks = [unsatChecks, 22];
    end

    % ------------------------------------------------------------
    % If both local checks are satisfied, the decoder sees no local
    % reason to update. But since this is a non-absorbing stochastic
    % model, we allow either:
    %
    %   1) stay in the same state with probability (1-epsilonRandom)
    %   2) random/noisy update with probability epsilonRandom
    %
    % This is what allows the correct state (2,7) to be left.
    % ------------------------------------------------------------

    if isempty(unsatChecks)

        % Stay due to no local unsatisfied checks
        P(i,i) = P(i,i) + (1 - epsilonRandom);

        % Random/noise-driven update
        P = add_random_update_with_probability(P, i, states, epsilonRandom);

        continue;
    end

    % ------------------------------------------------------------
    % If one or both local checks are unsatisfied, update either v26
    % or v31 with equal probability.
    % ------------------------------------------------------------

    vnList = [26, 31];

    for vn = vnList

        pChooseVN = 1/2;

        % --------------------------------------------------------
        % CN-guided candidate symbols from unsatisfied CNs
        % --------------------------------------------------------

        candidates = [];

        for c = unsatChecks

            if vn == 26

                % Solve:
                % h(i,26)*x26 + h(i,31)*x31 + Ri = 0
                %
                % x26 = (h(i,31)*x31 + Ri) / h(i,26)
                %
                % In GF(2^m), subtraction is same as addition.

                if c == 9
                    temp = gf_add(gf_mul(h931,x31), R9);
                    cand = gf_div(temp, h926);

                elseif c == 22
                    temp = gf_add(gf_mul(h2231,x31), R22);
                    cand = gf_div(temp, h2226);
                end

                newState = [cand, x31];

            elseif vn == 31

                % Solve:
                % h(i,26)*x26 + h(i,31)*x31 + Ri = 0
                %
                % x31 = (h(i,26)*x26 + Ri) / h(i,31)

                if c == 9
                    temp = gf_add(gf_mul(h926,x26), R9);
                    cand = gf_div(temp, h931);

                elseif c == 22
                    temp = gf_add(gf_mul(h2226,x26), R22);
                    cand = gf_div(temp, h2231);
                end

                newState = [x26, cand];
            end

            candidates = [candidates; newState];
        end

        candidates = unique(candidates, 'rows');

        % --------------------------------------------------------
        % Guided update probability
        % --------------------------------------------------------

        pGuided = pChooseVN * (1 - epsilonRandom);

        for r = 1:size(candidates,1)
            j = find_state_index(states, candidates(r,:));
            P(i,j) = P(i,j) + pGuided / size(candidates,1);
        end

        % --------------------------------------------------------
        % Random/noise-driven update probability
        % --------------------------------------------------------

        pRandom = pChooseVN * epsilonRandom;

        for sym = 0:7

            if vn == 26
                randomState = [sym, x31];
            else
                randomState = [x26, sym];
            end

            j = find_state_index(states, randomState);
            P(i,j) = P(i,j) + pRandom / 8;
        end
    end
end

%% ------------------------------------------------------------------------
% Check transition matrix validity
%% ------------------------------------------------------------------------

rowSums = sum(P,2);
fprintf('\nMaximum row-sum error = %.3e\n', max(abs(rowSums - 1)));

%% ------------------------------------------------------------------------
% Select initial state
%
% Example:
% Dominant observed transition:
%   true:  (2,7)
%   wrong: (1,3)
%
% So use initialState = [1,3].
%% ------------------------------------------------------------------------

initialState = [1, 3];

initialIndex = find_state_index(states, initialState);

p0 = zeros(1,numStates);
p0(initialIndex) = 1;

%% ------------------------------------------------------------------------
% Probability distribution after T iterations
%% ------------------------------------------------------------------------

pT = p0 * (P^T);

fprintf('\nInitial state = (%d,%d)\n', initialState(1), initialState(2));
fprintf('Correct state = (%d,%d)\n', correctSymbolPair(1), correctSymbolPair(2));
fprintf('Probability of being in correct state after T=%d iterations = %.6f\n', ...
    T, pT(correctState));

%% ------------------------------------------------------------------------
% Probability of being in the correct state over time
%% ------------------------------------------------------------------------

Tmax = 100;

p = p0;
probCorrectHistory = zeros(Tmax,1);

for t = 1:Tmax
    p = p * P;
    probCorrectHistory(t) = p(correctState);
end

figure;
plot(1:Tmax, probCorrectHistory, 'o-', 'LineWidth', 1.5);
xlabel('Iteration number T');
ylabel(sprintf('P{X_T = (%d,%d)}', correctSymbolPair(1), correctSymbolPair(2)));
title(sprintf('Non-absorbing Markov correction probability from X_0 = (%d,%d)', ...
    initialState(1), initialState(2)));
grid on;

fprintf('\nFinite-time probabilities:\n');
for Tcheck = [1 2 3 5 10 20 23 30 50 100]
    fprintf('T = %3d, P_correct = %.6f\n', ...
        Tcheck, probCorrectHistory(Tcheck));
end

%% ------------------------------------------------------------------------
% Probability of being correct after T from every possible initial state
%% ------------------------------------------------------------------------

PT = P^T;
probCorrectFromEachState = PT(:,correctState);

resultTable = table(states(:,1), states(:,2), probCorrectFromEachState, ...
    'VariableNames', {'x26_hat_initial','x31_hat_initial','P_correct_after_T'});

disp(resultTable);

writetable(resultTable, 'nonabsorbing_GF8_4cycle_probabilities.csv');

%% ------------------------------------------------------------------------
% Steady-state distribution
%
% Since this is non-absorbing, the chain may converge to a steady-state.
% This tells you the long-run probability of being in the correct state.
%% ------------------------------------------------------------------------

[V,D] = eig(P');

eigenValues = diag(D);
[~,ind] = min(abs(eigenValues - 1));

pi_ss = real(V(:,ind));
pi_ss = pi_ss / sum(pi_ss);

fprintf('\nApproximate steady-state probability of correct state (%d,%d) = %.6f\n', ...
    correctSymbolPair(1), correctSymbolPair(2), pi_ss(correctState));

steadyTable = table(states(:,1), states(:,2), pi_ss, ...
    'VariableNames', {'x26_hat','x31_hat','steady_state_probability'});

writetable(steadyTable, 'nonabsorbing_GF8_4cycle_steady_state.csv');

figure;
bar(pi_ss);
xlabel('State index');
ylabel('Steady-state probability');
title('Steady-state distribution over 64 GF(8) local states');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function c = gf8_mul(a,b,prim_poly)
    % Multiply two GF(8) elements using polynomial basis.
    c = 0;
    aa = a;
    bb = b;

    while bb > 0
        if bitand(bb,1)
            c = bitxor(c,aa);
        end

        bb = bitshift(bb,-1);
        aa = bitshift(aa,1);

        % Reduce if degree >= 3
        if bitand(aa,8)
            aa = bitxor(aa,prim_poly);
        end
    end

    c = bitand(c,7);
end

function inva = gf8_inv(a,prim_poly)
    % Multiplicative inverse in GF(8)

    if a == 0
        error('Division by zero in GF(8).');
    end

    for x = 1:7
        if gf8_mul(a,x,prim_poly) == 1
            inva = x;
            return;
        end
    end

    error('No inverse found.');
end

function c = gf8_div(a,b,prim_poly)
    % Division in GF(8): a / b = a * inv(b)

    if b == 0
        error('Division by zero in GF(8).');
    end

    invb = gf8_inv(b,prim_poly);
    c = gf8_mul(a,invb,prim_poly);
end

function idx = find_state_index(states, targetState)
    % Find row index of a given state

    idx = find(states(:,1)==targetState(1) & ...
               states(:,2)==targetState(2), 1);

    if isempty(idx)
        error('State (%d,%d) not found.', targetState(1), targetState(2));
    end
end

function P = add_random_update_with_probability(P, i, states, pTotalRandom)
    % Random/noise-driven update from state i.
    % With total probability pTotalRandom:
    %   choose either VN with probability 1/2
    %   choose one of 8 GF(8) symbols uniformly

    currentState = states(i,:);
    x26 = currentState(1);
    x31 = currentState(2);

    for vn = [26,31]

        for sym = 0:7

            if vn == 26
                newState = [sym, x31];
            else
                newState = [x26, sym];
            end

            j = find_state_index(states, newState);

            P(i,j) = P(i,j) + pTotalRandom * (1/2) * (1/8);
        end
    end
end