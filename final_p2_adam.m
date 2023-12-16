function final_p2_adam()

A = readmatrix('Adjacency_matrix.csv');
N = size(A,1);
x = randn(N,1);
y = randn(N,1);
w = vertcat(x,y);

g = -1 * forces(x,y,A);
tol = 1e-5;
iter_max = 1e5;
f_values = zeros(iter_max,1);

% Initialization for Adam
eta = 0.001;
b1 = 0.9;
b2 = 0.999;
m = zeros(2*N,1);
v = zeros(2*N,1);

for k = 1:iter_max
    m = b1 * m + (1-b1) * g;
    v = b2 * v + (1-b2) * g .* g;
    mk = m/(1-b1^k);
    vk = v/(1-b2^k);
    w = w - eta * ones(length(2*N),1) ./ (sqrt(vk) + 1e-8) .* mk;
    x = w(1:N);
    y = w(N+1:2*N);
    u = U(x,y,A);
    g = -1 * forces(x,y,A);
    g_norm = norm(g);
    f_values(k) = g_norm;
    if mod(k,1e3) == 0
        fprintf('k = %d, u = %d, ||g|| = %d\n',k,u,g_norm);
    end
    if g_norm < tol
        break;
    end
end

figure(1);
fsz = 20;
f_d = log(f_values);
plot(f_d,'Linewidth',2);
xlabel('Iteration #','FontSize',fsz);
ylabel('Norm of Force in Log Scale','FontSize',fsz);

plot_graph(x,y,A)

end

function u = U(x,y,A)
    u = 0;
    for i = 1:size(A,1)
        for j = 1:size(A,1)
            if A(i,j) == 1
                u = u + (sqrt((x(i)-x(j))^2 + (y(i)-y(j))^2) - 1)^2;
            else
                tmp = min(0, sqrt((x(i)-x(j))^2 + (y(i)-y(j))^2) - sqrt(3));
                u = u + tmp^2;
            end
        end
    end
    u = 0.5 * u;
end