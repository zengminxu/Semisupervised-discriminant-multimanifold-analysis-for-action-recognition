function [out] = SPG(opts)
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

%% the SPG parameters
M=10;
GAMMA=10^(-4);
ALPHA_min=10^(-15);
ALPHA_max=10^(15);
sigma_1=0.1;
sigma_2=0.9;
ALPHA=1;

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

W = rand(dim,class_number);    % project matrix(classifier)
F = rand(n,class_number);      % predict labels;
Lwb = Lw-opts.beta*Lb;
iter = 1;
%% ----------------------------------------
% the objective function w.s.t. (F,W)
objective=trace(F'*Lwb*F) + trace((F-Y)'*(F-Y)) + opts.mu*trace((X'*W-F)'*(X'*W-F)) + opts.mu*opts.alpha*trace(W'*W);
Lk(1)=objective;
%% the derivative of the f(F,W) w.s.t. F and W   
gf = (Lwb+Lwb')*F + 2*(F-Y) - 2*opts.mu*(X'*W-F);
gw = 2*opts.mu*X*(X'*W-F) + 2*opts.mu*opts.alpha*W;
Error(iter)=sqrt(norm(gf,'fro')^2+norm(gw,'fro')^2);
%% ---------------------------------------------------
tic
while 1  
    %------------------------------------
    D_f=-ALPHA*gf;
    D_w=-ALPHA*gw;
    %---------------------------------------------
    lamda=1;
    F_new=F+lamda*D_f;
    W_new=W+lamda*D_w;
    %---------------------------------------------
    %%  Calculate the inequality of the algorithm SPG %%
    jk=min(iter,M);
    Lj0=max(Lk(iter+1-jk:iter));
    Ljk=Lj0+GAMMA*lamda*(trace(D_f'*gf)+trace(D_w'*gw))+abs(objective)/iter^(1.1); 
    Lk_new=trace(F_new'*Lwb*F_new) + trace((F_new-Y)'*(F_new-Y)) + opts.mu*trace((X'*W_new-F_new)'*(X'*W_new-F_new)) + opts.mu*opts.alpha*trace(W_new'*W_new);
    while (Lk_new>Ljk)&&(lamda>10^(-15)) 
        lamda=(sigma_1*lamda+sigma_2*lamda)/2;
        F_new=F+lamda*D_f;
        W_new=W+lamda*D_w;
        Ljk=Lj0+GAMMA*lamda*(trace(D_f'*gf)+trace(D_w'*gw)); 
        Lk_new=trace(F_new'*Lwb*F_new) + trace((F_new-Y)'*(F_new-Y)) + opts.mu*trace((X'*W_new-F_new)'*(X'*W_new-F_new)) + opts.mu*opts.alpha*trace(W_new'*W_new);
    end
    
    %% ---------------------------------
    Lk(iter+1)=Lk_new;  % save new f(F,W)
    S_1=F_new-F;
    S_2=W_new-W;
    gf_new = (Lwb+Lwb')*F_new + 2*(F_new-Y) - 2*opts.mu*(X'*W_new-F_new);
    gw_new = 2*opts.mu*X*(X'*W_new-F_new) + 2*opts.mu*opts.alpha*W_new;
    Y_1=gf_new-gf;
    Y_2=gw_new-gw;
    b_k=trace(S_1'*Y_1)+trace(S_2'*Y_2);
    if b_k<=0
        ALPHA=ALPHA_max;
    else
        a_k=trace(S_1'*S_1)+trace(S_2'*S_2);
        ALPHA=min(ALPHA_max,max(ALPHA_min,a_k/b_k));
    end
    gf=gf_new; 
    gw=gw_new; 
    Error(iter)=sqrt(norm(gf,'fro')^2+norm(gw,'fro')^2);
    F=F_new; 
    W=W_new; 
    
    %-------------------------------------------------------------------
    reError(iter) = abs((Lk_new-Lk(iter))/Lk(iter));
    if  Error(iter) < opts.gtol || reError(iter) < opts.gtol || iter == opts.mxitr
        out.msg = 'converge';
        break;
    end        
    %-------------------------------------------------------------------
    ol1=sprintf('%3d',iter);
    ol2=sprintf('%14.2e',Lk(iter));
    ol3=sprintf('%14.2e',Error(iter));
    ol4=sprintf('%14.2e',reError(iter));
    ol=[ol1,'  ',ol2,'  ',ol3,'  ',ol4];
    disp(ol);
    
    iter=iter+1;    
end
if iter >= opts.mxitr
    out.msg = 'exceed max iteration';
end

out.Error = Error(iter);
out.reError = reError(iter);
out.fval = Lk;
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

% construct the Nearest neighbor graph between class
function W = constructWw(fea,labels,k)
% each row is a data
[nSmp nFea] = size(fea);
W = zeros(nSmp,nSmp);
class_num = max(labels);
Add = 0;
sigma = 1; 
for i = 1:class_num
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

% construct the Nearest neighbor graph between class
function W = constructWb(fea,labels,k)
% each row is a data
[nSmp nFea] = size(fea);
W = zeros(nSmp,nSmp);
class_num = max(labels);
for i = 1:class_num
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