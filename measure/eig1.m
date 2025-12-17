function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
% 计算一个矩阵的特征值和特征向量
% 输入参数：A 输入矩阵（通常为方阵）；
%          c 要求的特征值（特征向量）的数量。如果没有传入，则取默认值（矩阵的行数）
%          isMax：布尔值，用于决定是否选择最大特征值。默认值为 1（即选择最大特征值）
%          isSym：布尔值，指示是否对称。默认值为 1（意味着将输入矩阵对称处理）。
% 输出参数: eigvec：特征向量
%          eigval：所选的特征值
%          eigval_full：完整的特征值列表

if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
end;
[v d] = eig(A);
d = diag(d);
%d = real(d);
if isMax == 0
    [d1, idx] = sort(d);
else
    [d1, idx] = sort(d,'descend');
end;

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);