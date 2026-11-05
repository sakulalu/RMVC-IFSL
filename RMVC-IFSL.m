function [U,A,C,D,G,F,obj,label] = double_self_paced_with_rank(X,Y,numanchor,d,lambda,mu)

%% basic information
numsample = size(Y,1);
m = numanchor;
numclass = length(unique(Y));
numview = length(X);

%% normalization
for i = 1:numview
    X{i} = mapstd(X{i}',0,1);   % d_i × n
end

%% initialization
gamma_SPLs = 0.2;
gamma_SPLb = 0.2;

E_L_f = cell(numview,1);        % feature-level weights
E_R_s = ones(numsample,1);      % sample-level weights

U = cell(numview,1);
A = zeros(d,m);
C = zeros(m,numsample);
C(:,1:m) = eye(m);

D = cell(numview,1);
G = eye(m,numclass);
F = eye(numclass,numsample);

for i = 1:numview
    di = size(X{i},1);
    E_L_f{i} = ones(di,1);
    U{i} = zeros(di,d);
    D{i} = zeros(di,numsample);
end

flag = 1;
iter = 0;
maxIter = 50;
IterMax = 50;

while flag
    iter = iter + 1;

    %% current squared weights
    s2 = E_R_s.^2;              % n × 1
    f2 = cell(numview,1);       % each f2{i}: d_i × 1
    for i = 1:numview
        f2{i} = E_L_f{i}.^2;
    end

    %% optimize Uv
    for i = 1:numview
        % equivalent to: Ef2{i}*(X{i}-D{i})*Es2
        X_weighted = (f2{i} .* (X{i} - D{i})) .* s2';
        temp_Uv = X_weighted * C' * A';

        [W, ~, V] = svd(temp_Uv, 'econ');
        U{i} = W * V';
    end

    %% optimize A
    A_part = 0;
    for i = 1:numview
        % equivalent to: U{i}'*Ef2{i}*(X{i}-D{i})*Es2*C'
        X_weighted = (f2{i} .* (X{i} - D{i})) .* s2';
        temp_A = U{i}' * X_weighted * C';
        A_part = A_part + temp_A;
    end

    [Unew, ~, Vnew] = svd(A_part, 'econ');
    A = Unew * Vnew';

    %% optimize Dv
    denom_D = mu + rms(s2);

    for i = 1:numview
        residual_D = X{i} - U{i} * A * C;

        % equivalent to: Ef2{i}*(X{i}-U{i}*A*C)*Es2
        H = (f2{i} .* residual_D) .* s2';

        for ii = 1:numsample
            ut = H(:,ii) ./ denom_D;
            D{i}(:,ii) = EProjSimplex_new(ut);
        end
    end

    %% optimize C
    B = zeros(numsample, m);
    sumEf2 = 0;

    for i = 1:numview
        % equivalent to: Es2*(X{i}-D{i})'*Ef2{i}
        X_weighted = (f2{i} .* (X{i} - D{i})) .* s2';
        B = B + X_weighted' * U{i} * A;

        sumEf2 = sumEf2 + rms(f2{i});
    end

    Graph = zeros(numsample, m);
    Graph(1:m, :) = eye(m);

    % co-clustering
    [label, ~, P, ~, ~, term] = coclustering_bipartite_fast1( ...
        B, Graph, numclass, sumEf2 + lambda + rms(s2), IterMax);

    C = P';

    %% update s
    lsi = zeros(1, numsample);

    for i = 1:numview
        % equivalent to: diag(E_L_f{i})*(X{i}-U{i}*A*C-D{i})
        residual_s = E_L_f{i} .* (X{i} - U{i} * A * C - D{i});
        lsi = lsi + sum(residual_s.^2, 1);
    end

    lsi = mapminmax(lsi, 0, 1);

    me = 1 + exp(-gamma_SPLs);
    for i = 1:numsample
        de = 1 + exp(lsi(i) - gamma_SPLs);
        E_R_s(i) = sqrt(me / de);
    end
    gamma_SPLs = 1.1 * gamma_SPLs;

    %% optimize fv
    me = 1 + exp(-gamma_SPLb);

    for i = 1:numview
        di = size(X{i},1);

        residual_f = (X{i} - U{i} * A * C - D{i}) * E_R_s;
        lfi = sum(residual_f.^2, 2);
        lfi = mapminmax(lfi', 0, 1);

        for j = 1:di
            de = 1 + exp(lfi(j) - gamma_SPLb);
            E_L_f{i}(j) = sqrt(me / de);
        end
    end

    gamma_SPLb = 1.1 * gamma_SPLb;

    %% optimize G
    [Ug, ~, Vg] = svd(C * F', 'econ');
    G = Ug * Vg';

    %% optimize F
    C_norm = sum(C.^2, 1);          % 1 × n
    G_norm = sum(G.^2, 1)';         % k × 1
    Dist = G_norm + C_norm - 2 * (G' * C);   % k × n

    [~, idx] = min(Dist, [], 1);

    F = zeros(numclass, numsample);
    F(sub2ind(size(F), idx, 1:numsample)) = 1;

    %% objective
    term1 = 0;
    for i = 1:numview
        residual_obj = X{i} - U{i} * A * C - D{i};

        % equivalent to:
        % diag(E_L_f{i}) * residual_obj * diag(E_R_s)
        residual_weighted = (E_L_f{i} .* residual_obj) .* E_R_s';

        term1 = term1 + norm(residual_weighted, 'fro') + mu * norm(D{i}, 'fro');
    end

    term2 = lambda * norm(C - G * F, 'fro');
    obj(iter) = term1 + term2;

    %% stopping criterion
    if (iter > 1) && ...
       (abs((obj(iter - 1) - obj(iter)) / obj(iter - 1)) < 1e-4 ...
       || iter > maxIter ...
       || obj(iter) < 1e-10)
        flag = 0;
    end
end
end
