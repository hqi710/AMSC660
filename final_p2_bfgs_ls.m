function final_p2_bfgs_ls()

A = readmatrix('Adjacency_matrix.csv');
N = size(A,1);
x = randn(N,1);
y = randn(N,1);

u = U(x,y,A);
f = forces(x,y,A);
f_norm = norm(f);
tol = 1e-3;
iter = 1;
iter_max = 1e3;

% Initialization for BFGS
B = eye(2*N);
xk = x;
yk = y;
fk = f;

% parameters for backtracking line search
fail_flag = 0;
c = 0.75;
rho = 0.9;

while f_norm > tol && iter < iter_max
    [~,flag] = chol(B);
    if flag == 0 % B is SPD, use BFGS direction
        p = B\f;
        dir = "BFGS";
    else % use the steepest descent direction
        p = f;
        dir = "SD";
    end

    % normalize the search direction if its length greater than 1
    norm_p = norm(p);
    if norm_p > 1
        p = p/norm_p;
    end

    % do backtracking line search along the direction p
    a = 1;
    u_temp = U(x + a*p(1:N), y + a*p(N+1:2*N), A);
    cpf = -c*p'*f;
    while u_temp > u + a*cpf % check Wolfe's condition 1
        a = a*rho;
        if a < 1e-14
            fprintf("line search failed\n");
            iter = iter_max;
            fail_flag = 1;
            break;
        end
        u_temp = U(x + a*p(1:N), y + a*p(N+1:2*N), A);        
    end

    x = x + a*p(1:N);
    y = y + a*p(N+1:2*N);
    u = U(x,y,A);
    f = forces(x,y,A);
    norm_f = norm(f);

    if mod(iter, 20) == 0
        B = eye(2*N);
    else
        s = vertcat(x - xk, y - yk);
        q = -f + fk;
        B = B - (B*s*s'*B) / (s'*B*s) + q*q'/(q'*s);
    end

    xk = x;
    yk = y;
    fk = f;

    fprintf("iter %d : dir = %s, u = %d, ||grad u|| = %d, step length = %d\n",iter,dir,u,norm_f,a);
    
    if fail_flag == 1
        break;
    end

    iter = iter + 1;
end

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