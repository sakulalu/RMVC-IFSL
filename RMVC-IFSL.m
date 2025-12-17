function [U,A,C,D,G,F,obj,label] = double_self_paced_with_rank(X,Y,numanchor,d,lambda,mu)

%% 
numsample = size(Y,1);
m = numanchor;
numclass = length(unique(Y));
numview = length(X);

%% 
for i = 1 : numview
    X{i} = mapstd(X{i}',0,1);
end

%% Initilize
gamma_SPLs=0.2;
gamma_SPLb=0.2;
E_L_f = cell(numview,1);
E_R_s = ones(numsample,1);
U = cell(numview,1);
A = zeros(d,m);
C = zeros(m,numsample);
C(:, 1:m) = eye(m);
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
maxIter = 1;
IterMax = 50;

while flag
    iter = iter + 1;
    
    %% optimize Uv
    clear Ef2;
    Ef2 = cell(numview,1);
    Es2 = diag(E_R_s.^2);
    for i = 1:numview
        Ef2{i} = diag(E_L_f{i}.^2);
        temp_Uv = Ef2{i}*(X{i}-D{i})*Es2*C'*A';
        [W, ~, V] = svd(temp_Uv, 'econ');
        U{i} = W * V';
    end

    %% optimize A
    A_part = 0;
    for i = 1:numview
        temp_A = U{i}'*Ef2{i}*(X{i}-D{i})*Es2*C';
        A_part = A_part + temp_A; 
    end
    [Unew, ~, Vnew] = svd(A_part, 'econ');
    A = Unew * Vnew';
    
    %% optimize Dv
    for i = 1:numview
        H = Ef2{i}*(X{i} - U{i}*A*C)*Es2;      
        for ii = 1:numsample
            ut = H(:,ii)./(mu+rms(E_R_s.^2));
            D{i}(:,ii) = EProjSimplex_new(ut);
        end
    end

    %% optimize C
    B = zeros(numsample, m);
    sumEf2 = 0;
    for i = 1:numview
        B = B + Es2*(X{i}-D{i})'*Ef2{i}*U{i}*A;
        sumEf2 = sumEf2 + rms(E_L_f{i}.^2);
    end
    Graph = zeros(numsample, m);
    Graph(1:m, :) = eye(m);
    % 双聚类分析
    [label, ~, P, ~, ~, term] = coclustering_bipartite_fast1(B, Graph, numclass, sumEf2+lambda+rms(E_R_s.^2), IterMax);
    C = P';

    %% update s
    clear residual_s;
    residual_s = cell(numview,1);
    for i = 1:numview
        residual_s{i} = diag(E_L_f{i})*(X{i}-U{i}*A*C-D{i});
    end
    lsi = zeros(1,numsample);
    for j = 1:numsample
        temp_lsi = 0;
        for i = 1:numview
            di = size(X{i},1);
            temp_L = sum(residual_s{i}.^2,1);
            temp_lsi = temp_lsi + temp_L(j);
        end
        lsi(j) = temp_lsi;
    end
    lsi = mapminmax(lsi, 0, 1);
    me = 1 + exp(-1*gamma_SPLs);
    for i = 1:numsample
        de = 1 + exp(lsi(i)-gamma_SPLs);
        E_R_s(i) = sqrt(me/de);
    end
    gamma_SPLs =  1.1 * gamma_SPLs;

    %% optimize fv
    me = 1 + exp(-1*gamma_SPLb); 
    for i = 1:numview
        di = size(X{i},1);
        residual_f = (X{i} - U{i}*A*C - D{i})*E_R_s;
        lfi = sum(residual_f.^2,2);
        lfi = mapminmax(lfi', 0, 1);
        for j = 1:di
            de = 1 + exp(lfi(j)-gamma_SPLb);
            E_L_f{i}(j) = sqrt(me/de);
        end
    end
    gamma_SPLb =  1.1 * gamma_SPLb;

    %% optimize G  
    [Ug,~,Vg] = svd(C*F','econ');
    G = Ug*Vg';

    % optimize F
    F=zeros(numclass,numsample);
    for i=1:numsample
        Dis=zeros(numclass,1);
        for j=1:numclass
            Dis(j)=(norm(C(:,i)-G(:,j)))^2;
        end
        [~,r]=min(Dis);
        F(r(1),i)=1;
    end

    %%
    term1 = 0;
    for i = 1:numview
        term1 = term1+ norm(diag(E_L_f{i})*(X{i}-U{i}*A*C-D{i})*diag(E_R_s),'fro') + mu*norm(D{i},'fro');
    end
    term2 = lambda * norm(C-G*F,'fro');
    obj(iter) = term1 + term2;

    %%
    if (iter > 1) && (abs((obj(iter - 1) - obj(iter)) / (obj(iter - 1))) < 1e-4 || iter > maxIter || obj(iter) < 1e-10)
        flag = 0;
    end
end

