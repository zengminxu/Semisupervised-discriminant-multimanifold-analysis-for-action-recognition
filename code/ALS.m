function [out] = ALS(opts)
% input:
% X: IDT+FV or TDD+FV representation of features on JHMDB
% Y: groundtruth labels;
% k: number of nearest neighbors for Laplacian matrix
% alpha, beta, mu are semi-supervised parameters
% output: 
% out.Error: Error
% out.reError: relative Error
% out.fval: objective function value
% out.iter: iteration number;
% out.time: running time
% Reference:
% Semi-Supervised Discriminant Multi-Manifold Analysis for Action Recognition, TNNLS2019
 
X = opts.X;
Y = opts.Y;
trainLabels = opts.trainLabels;
testLabels = opts.testLabels;
k = opts.k;

class_number = size(Y,2);      % class_number number
t = rank(X);
%========================PCA========================
if issparse(X)
    [U1, Dx, V1] = svds(X, t);
else
    [U1, Dx, V1] = svd(X, 'econ');
end
fea = X;            
X = Dx*V1';         
[dim,n] = size(X);      % dim dimension x n samples
%=====================Laplacian====================
[Lw,Lb] = xzm_Laplacian_GK(fea, trainLabels, k);

%% ------------------------------
disp('_______________________________________________________');
disp('  k       Obji_value        Error        relative Error');
disp('_______________________________________________________');

W = rand(dim,class_number);    % project matrix(class_numberifier)
F = rand(n,class_number);      % predict labels;
I = eye(dim);
Lwb = Lw-opts.beta*Lb;
iter = 1;
obji = 1;
%========================ALS======================== 
tic
while 1      
    objective(iter) = trace(F'*Lwb*F)+trace((F-Y)'*(F-Y))+opts.mu*trace((X'*W-F)'*(X'*W-F))+opts.mu*opts.alpha*trace(W'*W);   % xzm
    reError(iter) = abs((objective(iter)-obji)/obji);
    obji = objective(iter);  
    
    gf = (Lwb+Lwb')*F + 2*(F-Y) - 2*opts.mu*(X'*W-F);
    gw = 2*opts.mu*X*(X'*W-F) + 2*opts.mu*opts.alpha*W; 
    Error(iter) = sqrt(norm(gf,'fro')^2+norm(gw,'fro')^2); 
    %-------------------------------------------------------------------    
    if  Error(iter) < opts.gtol || reError(iter) < opts.gtol || iter == opts.mxitr
        out.msg = 'converge';
        break;
    end     
    %-------------------------------------------------------------------  
    ol1=sprintf('%3d',iter);
    ol2=sprintf('%14.2e',objective(iter));
    ol3=sprintf('%14.2e',Error(iter));
    ol4=sprintf('%14.2e',reError(iter));
    ol=[ol1,'  ',ol2,'  ',ol3,'  ',ol4];
    disp(ol);    
    
%     F = inv(Lwb+I+opts.mu*I)*(Y+opts.mu*X'*W);
%     W = inv(opts.alpha*I+X*X')*F;
    F = (Lwb+I+opts.mu*I)\(Y+opts.mu*X'*W);   % speed up
    W = (opts.alpha*I+X*X')\F;   
    
    iter = iter+1;
end
if iter >= opts.mxitr
    out.msg = 'exceed max iteration';
end

out.Error = Error(iter);
out.reError = reError(iter);
out.fval = objective;
out.iter = iter;
out.time=toc;
end

function [Lw Lb] = xzm_Laplacian_GK(fea, trainLabels, k)
% each column is a data
W = constructWw(fea',trainLabels,k);
D = diag(sum(W));
Aw = W;
Lw = D-W;
W = constructWb(fea',trainLabels,k);
D = diag(sum(W));
Ab = W;
Lb = D-W;
end

% construct the Nearest neighbor graph between class_number
function W = constructWw(fea,labels,k)
% each row is a data
[nSmp nFea] = size(fea);
W = zeros(nSmp,nSmp);
class_number_num = max(labels);
Add = 0;
sigma = 1; 
for i = 1:class_number_num
    feabc = fea(find(labels==i)',:); 
    dist = EuDist2(feabc,feabc); 
    [dump idx] = sort(dist,2); % sort each row
    idx = idx(:,2:k+1);        % k neighbor samples by Euclidean distance 
    dump = dump(:,2:k+1);      % the Euclidean distance of k neighbor samples  
    for m = 1:size(idx,1)
        for n = 1:size(idx,2)            
            W(Add+m,Add+idx(m,n)) = 1;
        end
    end
    Add = size(find(labels<=i),2);
end
W = (W + W')./2;
end

% construct the Nearest neighbor graph between class_number
function W = constructWb(fea,labels,k)
% each row is a data
[nSmp nFea] = size(fea);
W = zeros(nSmp,nSmp);
class_number_num = max(labels);
for i = 1:class_number_num
    feabc = fea(find(labels==i)',:); 
    featemp = fea;
    featemp(find(labels==i)',:) = fea(find(labels==i)',:) - 1e+6;
    dist = EuDist2(feabc,featemp);
    [dump idx] = sort(dist,2); % sort each row
    idx = idx(:,1:k);          % k neighbor samples by Euclidean distance
    dump = dump(:,1:k);        % the Euclidean distance of k neighbor samples
    Add1 = size(find(labels<=i-1),2);
    for m = 1:size(idx,1)
        for n = 1:size(idx,2)            
            W(m+Add1,idx(m,n)) = 1;
        end
    end
end
W = (W + W')./2;
end