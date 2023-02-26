% 4th b question solution
N_arr=[20,50,100];
threshold = 10^-3;
sigma_array = [0.001,0.01];
noise = 1;
for sigma=sigma_array
    figure;
    for N_id=1:size(N_arr,2)
        N = N_arr(:,N_id);
        M_arr = 10:5:100;
        S_arr = 1:3:floor(N/2);
        prob_plot = zeros(size(S,2),size(M,2));
        norm_error_plot = zeros(size(S,2),size(M,2));
        for M=M_arr
            for s_max = S_arr
                disp([N,M,s_max]);
                M_id = find(M_arr==M);
                S_id = find(S_arr==s_max);
                [norm_error_plot(M_id,S_id),prob_plot(M_id,S_id)] = experiment_noise2(N,M,s_max,sigma,threshold,noise,T);
            end
        end
        subplot(3,2,(N_id-1)*2+1);
        imshow(norm_error_plot,'InitialMagnification',100000);
        colormap('winter');
        colorbar;
        xlabel('s-max');
        ylabel('M');
        title(strcat('Normalized Error Phase Transition Plot for sd=',string(sigma),' N=',string(N)));
        subplot(3,2,(N_id-1)*2+2);
        imshow(prob_plot,'InitialMagnification',100000);
        colormap('winter');
        colorbar;
        xlabel('s-max');
        ylabel('M');
        title(strcat('Probability of Success Phase Transition Plot for sd=',string(sigma),' N=',string(N)));
    end
end


function [avg_norm_error,avg_prob] = experiment_noise2(N,M,s_max,sigma,threshold,noise,T)
    avg_norm_error = 0;
    avg_prob = 0;
    parfor i = 1:T
        [norm_error,prob] = monteCarlo_noise2(N,M,s_max,sigma,threshold,noise);
        avg_norm_error = avg_norm_error + norm_error;
        avg_prob = avg_prob + prob;
    end
    avg_norm_error = avg_norm_error/T;
    avg_prob = avg_prob/T;
end

function [norm_error,prob] = monteCarlo_noise2(N,M,s_max,sigma,threshold,noise)
    if s_max > M
        norm_error = 1;
        prob = 0;
    else
        x = generateX(N,s_max);
        n = genarateNoise(sigma,M,noise);
        A = generateA(M,N,sigma);
        y = generateSignal(A,x,n);
        x_hat = OMP_noise2(A,y,n);
        norm_error = normalizedError(x,x_hat);
        prob = norm_error < threshold;
    end
end

function x_hat = OMP_noise2(A,y,n)
    [M,N] = size(A);
    if M ~= size(y)
        error('Dimension of y and A not matched');
    end
    r = y;
    x_hat = zeros(N,1);
    lambdas = zeros(1,N);
    k = 0;
    while l2Norm(r) > l2Norm(n)
        k=k+1;
        [~,lambdas(:,k)] = max(abs(innerProduct(A,r)));
        max_inner_product_sub_space = A(:,lambdas(:,1:k));
        normalEqnSol= findNormalEquation(max_inner_product_sub_space);
        x_hat(lambdas(:,1:k),:) = normalEqnSol*y;
        r = y - A*x_hat;
    end
end

function soln= findNormalEquation(A)
    soln = inv(transpose(A)*A)*transpose(A);
end

function norm_error = normalizedError(x_true,x_approx)
    norm_error = l2Norm(x_true - x_approx)/l2Norm(x_true);
end

function y = generateSignal(A,x,n)
    y = A*x+n;
end

function A = generateA(M,N,sigma)
    A = normrnd(0,sigma,[M,N]);
    A = normalize(A);
end

function x = generateX(N,s)
    x_index = randsample(1:N,s);
    x = zeros(1,N);
    x(:,x_index) = datasample([-1,1],s).*ceil(rand(1,s)*10);
    x = x';
end

function n = genarateNoise(sigma,M,noise)
    n = normrnd(0,sigma,[M,1]);
    n=n*noise;
end

function normalized_A = normalize(A)
    normalized_A = A./l2Norm(A);
end

function l = l2Norm(x)
    l = sqrt(innerProduct(x,x));
end

function dot_product = innerProduct(x,y)
    dot_product = sum(x.*y,1);
end