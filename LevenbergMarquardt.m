function [w,fvals,ngvals] = LevenbergMarquardt(fun,r_and_J,w,kmax,tol)

%% parameters for trust region
Delta_max = 5; % the max trust-region radius
Delta_min = 1e-12; % the minimal trust-region radius
Delta = 1; % the initial radius
eta = 0.1; % step rejection parameter
subproblem_iter_max = 5; % the max # of iteration for quadratic subproblems
tol_sub = 1e-1; % relative tolerance for the subproblem
rho_good = 0.75;
rho_bad = 0.25;

iter = 1;
f = fun(w);
fvals = zeros(kmax + 1,1);
fvals(1) = f;
ngvals = zeros(kmax + 1,1);
[r, J] = r_and_J(w);
g = J'*r;
norm_g = norm(g);
ngvals(1) = norm_g;
I = eye(length(w));

while norm_g > tol && iter < kmax
    % solve the constrained minimization problem
    [r, J] = r_and_J(w);
    B = J'*J + 1e-6*eye(size(J,2));
    flag_boundary = 0;
    j_sub = 0;
    p = -B\g;
    p_norm = norm(p);
    if p_norm > Delta % else: we are done with solving the subproblem
        flag_boundary = 1;
    end
    if flag_boundary == 1 % solution lies on the boundary
        lambda = 1;
        R = chol(B+lambda*I);
        flag_subproblem_success = 0;
        while j_sub < subproblem_iter_max
            j_sub = j_sub + 1;
            p = -R\(R'\g);
            p_norm = norm(p);
            dd = abs(p_norm - Delta);
            if dd < tol_sub*Delta
                flag_subproblem_success = 1;
                break
            end
            q = R'\p;
            q_norm = norm(q);
            dlambda = ((p_norm/q_norm)^2)*(p_norm - Delta)/Delta;
            lambda_new = lambda + dlambda;
            if lambda_new > 0
                lambda = lambda_new;
            else
                lambda = 0.5*(lambda);
            end
            R = chol(B+lambda*I);
        end
        if flag_subproblem_success == 0
            p = cauchy_point(B,g,Delta);
        end
    end

    % assess the progress
    wnew = w + p;
    fnew = fun(wnew);
    [rnew, Jnew] = r_and_J(wnew);
    gnew = Jnew'*rnew;
    mnew = f + p'*g + 0.5*p'*B*p;
    rho = (f - fnew+1e-14)/(f - mnew+1e-14);

    % adjust the trust region
    if rho < rho_bad
        Delta = max([0.25*Delta,Delta_min]);
    else
        if rho > rho_good && flag_boundary == 1
            Delta = min([Delta_max,2*Delta]);
        end
    end
    
    % accept or reject step
    if rho > eta            
        w = wnew;
        f = fnew;
        g = gnew;
        norm_g = norm(g);
        fprintf('Accept: iter # %d: f = %.10f, |df| = %.4e, rho = %.4e, Delta = %.4e, j_sub = %d\n',iter,f,norm_g,rho,Delta,j_sub);
    else
        fprintf('Reject: iter # %d: f = %.10f, |df| = %.4e, rho = %.4e, Delta = %.4e, j_sub = %d\n',iter,f,norm_g,rho,Delta,j_sub);
    end
    iter = iter + 1;
    fvals(iter) = f;
    ngvals(iter) = norm_g;
end
end

%%
function p = cauchy_point(B,g,R)
    ng = norm(g);
    ps = -g*R/ng;
    aux = g'*B*g;
    if aux <= 0
        p = ps;
    else
        a = min(ng^3/(R*aux),1);
        p = ps*a;
    end
end
        
        
    
    
