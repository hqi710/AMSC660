function mnist_2categories_hypersurface()
close all
fsz = 20;
%% Pick the number of PCAs for the representation of images
nPCA = 3;
%%
mdata = load('mnist.mat');
imgs_train = mdata.imgs_train;
imgs_test = mdata.imgs_test;
labels_test = mdata.labels_test;
labels_train = mdata.labels_train;
%% find 1 and 7 in training data
ind1 = find(double(labels_train)==2);
ind2 = find(double(labels_train)==8);
n1train = length(ind1);
n2train = length(ind2);
fprintf("There are %d 1's and %d 7's in training data\n",n1train,n2train);
train1 = imgs_train(:,:,ind1);
train2 = imgs_train(:,:,ind2);
%% find 1 and 7 in test data
itest1 = find(double(labels_test)==2);
itest2 = find(double(labels_test)==8);
n1test = length(itest1);
n2test = length(itest2);
fprintf("There are %d 1's and %d 7's in test data\n",n1test,n2test);
test1 = imgs_test(:,:,itest1);
test2 = imgs_test(:,:,itest2);
% %% plot some data from category 1
% figure; colormap gray
% for j = 1:20
%     subplot(4,5,j);
%     imagesc(train1(:,:,j));
%     axis off
% end
% %% plot some data from category 2
% figure; colormap gray
% for j = 1:20
%     subplot(4,5,j);
%     imagesc(train2(:,:,j));
%     axis off
% end
%% use PCA to reduce dimensionality of the problem to 20
[d1,d2,~] = size(train1);
X1 = zeros(n1train,d1*d2);
X2 = zeros(n2train,d1*d2);
for j = 1 : n1train
    aux = train1(:,:,j);
    X1(j,:) = aux(:)';
end
for j = 1 :n2train
    aux = train2(:,:,j);
    X2(j,:) = aux(:)';
end
X = [X1;X2];
D1 = 1:n1train;
D2 = n1train+1:n1train+n2train;
[U,Sigma,~] = svd(X','econ');
esort = diag(Sigma);
figure;
plot(esort,'.','Markersize',20);
grid;
Xpca = X*U(:,1:nPCA); % features
figPCA = figure; 
hold on; grid;
plot3(Xpca(D1,1),Xpca(D1,2),Xpca(D1,3),'.','Markersize',20,'color','k');
plot3(Xpca(D2,1),Xpca(D2,2),Xpca(D2,3),'.','Markersize',20,'color','r');
view(3)
%% split the data to training set and test set
Xtrain = Xpca;
Ntrain = n1train + n2train;
Xtest1 = zeros(n1test,d1*d2);
for j = 1 : n1test
    aux = test1(:,:,j);
    Xtest1(j,:) = aux(:)';
end
for j = 1 : n2test
    aux = test2(:,:,j);
    Xtest2(j,:) = aux(:)';
end
Xtest = [Xtest1;Xtest2]*U(:,1:nPCA);
%% category 1 (1): label 1; category 2 (7): label -1
label = ones(Ntrain,1);
label(n1train+1:Ntrain) = -1;
%% dividing hyperplane: w'*x + b
dim = nPCA;
%% optimize w and b using a smooth loss function and Stochastic optimizers
lam = 0.001; % Tikhonov regularization parameter
fun = @(I,w)qloss(I,Xtrain,label,w,lam);
gfun = @(I,w)qlossgrad(I,Xtrain,label,w,lam);
w = ones(dim^2+dim+1,1);
% params
frac = 100;
bsz = ceil(Ntrain/frac); % batch size
kmax = 1e3*frac; % the max number of iterations
tol = 1e-4;
% call the optimizer
% [w,f,gnorm] = SGD(fun,gfun,Xtrain,w,bsz,kmax,tol);
% [w,f,gnorm] = Nestorov(fun,gfun,Xtrain,w,bsz,kmax,tol);
[w,f,gnorm] = Adam(fun,gfun,Xtrain,w,bsz,kmax,tol);
d2 = dim^2;
W = reshape(w(1:d2),[dim,dim]);
v = w(d2+1:d2+dim);
b = w(end);
% plot the objective function
figure;
plot(f,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('f','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');
% plot the norm of the gradient
figure;
plot(gnorm,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('||g||','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');

%% apply the results to the test set
Ntest = n1test+n2test;
testlabel = ones(Ntest,1);
testlabel(n1test+1:Ntest) = -1;
qterm = diag(Xtest*W*Xtest');
test = testlabel.*qterm + ((testlabel*ones(1,dim)).*Xtest)*v + testlabel*b;
% test = testlabel.*(Xtest*wvec + b);
hits = find(test > 0);
misses = find(test < 0);
nhits = length(hits);
nmisses = length(misses);
fprintf('n_correct = %d, n_wrong = %d, accuracy %d percent\n',nhits,nmisses,nhits/Ntest);
end

%% The objective function
function f = qloss(I,Xtrain,label,w,lam)
f = sum(log(1 + exp(-myquadratic(Xtrain,label,I,w))))/length(I) + 0.5*lam*w'*w;
end

%% The gradient of the objective function
function g = qlossgrad(I,Xtrain,label,w,lam)
aux = exp(-myquadratic(Xtrain,label,I,w));
a = -aux./(1+aux);
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
ya = y.*a;
qterm = X'*((ya*ones(1,d)).*X);
lterm = X'*ya;
sterm = sum(ya);
g = [qterm(:);lterm;sterm]/length(I) + lam*w;
end

%%
function q = myquadratic(Xtrain,label,I,w)
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);
qterm = diag(X*W*X');
q = y.*qterm + ((y*ones(1,d)).*X)*v + y*b;
end

%% SGD
function [w,f,normgrad] = SGD(fun,gfun,Xtrain,w,bsz,kmax,tol)
alpha = 0.2;
n = size(Xtrain,1);
I = 1:n;
f = zeros(kmax + 1,1);
f(1) = fun(I,w);
normgrad = zeros(kmax,1);
for k = 1 : kmax
    Ig = randperm(n,bsz);
    b = gfun(Ig,w);
    normgrad(k) = norm(b);
    % w = w - alpha * b;
    w = w - alpha / sqrt(k) * b;
    f(k + 1) = fun(Ig,w);
    if mod(k,1000) == 0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
    end
    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end

%% Nestorov
function [w,f,normgrad] = Nestorov(fun,gfun,Xtrain,w,bsz,kmax,tol)
alpha = 0.01;
n = size(Xtrain,1);
I = 1:n;
f = zeros(kmax + 1,1);
f(1) = fun(I,w);
w_prev = w;
normgrad = zeros(kmax,1);
for k = 1 : kmax
    mu = 1 - 3/(5+k);
    y = w + mu*(w-w_prev);
    Ig = randperm(n,bsz);
    % disp(Ig)
    b = gfun(Ig,y);
    normgrad(k) = norm(b);
    w_prev = w;
    w = y - alpha * b;
    % w = w - alpha / k * b;
    f(k + 1) = fun(Ig,w);
    if mod(k,1000) == 0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
    end
    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end

%% Adam
function [w,f,normgrad] = Adam(fun,gfun,Xtrain,w,bsz,kmax,tol)
eta = 0.001;
b1 = 0.9;
b2 = 0.999;
m = zeros(length(w),1);
v = zeros(length(w),1);
n = size(Xtrain,1);
I = 1:n;
f = zeros(kmax + 1,1);
f(1) = fun(I,w);
normgrad = zeros(kmax,1);
for k = 1 : kmax
    Ig = randperm(n,bsz);
    g = gfun(Ig,w);
    normgrad(k) = norm(g);
    m = b1 * m + (1-b1) * g;
    v = b2 * v + (1-b2) * g .* g;
    mk = m/(1-b1^k);
    vk = v/(1-b2^k);
    w = w - eta * ones(length(w),1) ./ (sqrt(vk) + 1e-8) .* mk;
    f(k + 1) = fun(Ig,w);
    if mod(k,1000) == 0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
    end
    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end


