% 5th question solution
Data = load('/Users/abhijithdasharathi/Study/UCSD/Winter_23/ECE269/HW3/Data for HW3/Y1 Y2 Y3 and A1 A2 A3.mat');

% display compressed images
figure;
subplot(2,3,1);
image_y_1 = reshape(Data.y1,15,64);
imshow(image_y_1);
title('compressed image y1');

subplot(2,3,2);
image_y_2 = reshape(Data.y2,15,96);
imshow(image_y_2);
title('compressed image y2');

subplot(2,3,3);
image_y_3 = reshape(Data.y3,40,72);
imshow(image_y_3);
title('compressed image y3');

subplot(2,3,4);
image_y_1 = reshape(Data.y1,30,32);
imshow(image_y_1);
title('compressed image y1 as a qr code');

subplot(2,3,5);
image_y_2 = reshape(Data.y2,32,45);
imshow(image_y_2);
title('compressed image y2 as a qr code');

subplot(2,3,6);
image_y_3 = reshape(Data.y3,40,72);
imshow(image_y_3);
title('compressed image y3 as a qr code');

figure;
[~,s_max_1] = decodeX1(Data.A1,Data.y1,1);
[n_1,~] = decodeX2(Data.A1,Data.y1,s_max_1,2);
[~,~] = decodeX3(Data.A1,Data.y1,n_1,3);

[~,s_max_2] = decodeX1(Data.A2,Data.y2,4);
[n_2,~] = decodeX2(Data.A2,Data.y2,s_max_2,5);
[~,~] = decodeX3(Data.A2,Data.y2,n_2,6);
 
[~,s_max_3] = decodeX1(Data.A3,Data.y3,7);
[n_3,~] = decodeX2(Data.A3,Data.y3,s_max_3,8);
[~,~] = decodeX3(Data.A3,Data.y3,n_3,9);


function [n,s_max] = decodeX1(A,y,p)
    threshold = 10^-6;
    x_hat = OMP_noiseless(A,y,threshold);
    s_max = sum(x_hat~=0,1);
    n = y-A*x_hat;
    image_x = reshape(x_hat,90,160);
    subplot(3,3,p);
    imshow(image_x);
    title(strcat("Image decoded using noiseless OMP decoding with ||r||^2 < 10^-6 and s-max=",string(s_max)));
end

function [n,s_max] = decodeX2(A,y,s_max,p)
    x_hat = OMP_noise1(A,y,s_max);
    n = y-A*x_hat;
    image_x = reshape(x_hat,90,160);
    subplot(3,3,p);
    imshow(image_x);
    title(strcat("Image decoded using noisy OMP decoding with s-max = ",string(s_max)," known"));
end

function [n,s_max] = decodeX3(A,y,n,p)
    x_hat = OMP_noise2(A,y,n);
    s_max = sum(x_hat~=0,1);
    image_x = reshape(x_hat,90,160);
    subplot(3,3,p);
    imshow(image_x);
    title("Image decoded using noisy OMP decoding with ||r||^2 < ||n||^2 known");
end

function x_hat = OMP_noiseless(A,y,threshold)
    [M,N] = size(A);
    if M ~= size(y)
        error('Dimension of y and A not matched');
    end
    r = y;
    x_hat = zeros(N,1);
    lambdas = zeros(1,N);
    k = 0;
    while l2Norm(r) > threshold %&& k < s_max
        k=k+1;
        [~,lambdas(:,k)] = max(abs(innerProduct(A,r)));
        max_inner_product_sub_space = A(:,lambdas(:,1:k));
        normalEqnSol= findNormalEquation(max_inner_product_sub_space);
        x_hat(lambdas(:,1:k),:) = normalEqnSol*y;
        r = y - A*x_hat;
    end
end

function x_hat = OMP_noise1(A,y,s_max)
    [M,N] = size(A);
    if M ~= size(y)
        error('Dimension of y and A not matched');
    end
    r = y;
    x_hat = zeros(N,1);
    lambdas = zeros(1,s_max);
    k = 0;
    while k < s_max
        k=k+1;
        [~,lambdas(:,k)] = max(abs(innerProduct(A,r)));
        max_inner_product_sub_space = A(:,lambdas(:,1:k));
        normalEqnSol= findNormalEquation(max_inner_product_sub_space);
        x_hat(lambdas(:,1:k),:) = normalEqnSol*y;
        r = y - A*x_hat;
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