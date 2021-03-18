% William Ojemann
% Assignment 5
% AMATH 482
% 10 MAR 2021
clear; close all; clc;
%% Initial Processing - Do Not Run
% % % monte = read(VideoReader('monte_carlo_low.mp4'));
% % % ski = read(VideoReader('ski_drop_low.mp4'));
% % % [datam, m1, n1] = parametrize(monte);
% % % [datas, m2, n2] = parametrize(ski);
% % % save('Ski_data.mat','datas','m2','n2');
% % % save('Monte_data.mat','datam','m1','n1');
%% Loading
load('Monte_data.mat'); load('Ski_data.mat');
%% Monte Carlo Analysis
% num_components -- 20
% eig threshold  -- 0.5
% Background isolated in datalow_inter
data = datam; j = m1; k = n1;
t = size(data,2);
n = length(data);
X1 = data(:,1:end-1); X2 = data(:,2:end);
num_components = 20;
[U,Vs,l] = operator(X1,X2,num_components);
mu = diag(l);
omega = log(mu);
Phi = U*Vs;
Y0 = Phi\X1(:,1);
modes = zeros(length(Y0),t);
for i = 1:t
    modes(:,i) = Y0.*exp(omega*i);
end
dataDMD = Phi*modes;
omegano = abs(omega);
figure(2)
thresh = 0.5;
I = find(omegano < thresh);
plot(omegano,'o');
hold on
plot([0 length(omegano)],[thresh thresh],'-k');
ylim([-.1 (thresh + .1)])
title('Monte Carlo Eigenvalues','Fontsize',16);
ylabel('Magnitude','Fontsize',14);
xlabel('Index','Fontsize',14);
datalow_inter = Phi(:,I)*modes(I,:);
datasparse_inter = data - abs(datalow_inter);
rcheck = datasparse_inter < 0;
R = datasparse_inter.*rcheck;
datalow = abs(datalow_inter);
datasparse = datasparse_inter-R;
figure(100)
i = 1;
subplot(3,1,i)
imshow(reshape(data(:,i),j,k));
subplot(3,1,i+1)
imshow(reshape(datalow(:,i),j,k));
subplot(3,1,i+2)
imshow(reshape(datasparse(:,i),j,k));
sgtitle('Monte Carlo Video Decomposed','Fontsize',16)
%% Ski Drop Analysis
% num_components -- 20
% eig threshold  -- 0.05
% add abs(min) for sparse instead of R
% don't add R for low rank
data = datas; j = m2; k = n2;
t = size(data,2);
n = length(data);
X1 = data(:,1:(end-1)); X2 = data(:,2:end);
num_components = 20;
[U,Vs,l] = operator(X1,X2,num_components);
mu = diag(l);
omega = log(mu);
Phi = U*Vs;
Y0 = Phi\X1(:,1);
modes = zeros(length(Y0),t);
for i = 1:t
    modes(:,i) = Y0.*exp(omega*i);
end
dataDMD = Phi*modes;
omegano = abs(omega);
figure(1)
thresh = 0.05;
I = find(omegano < thresh);
plot(omegano(I),'bo'); drawnow
hold on
plot([0 length(omegano)],[thresh thresh],'-k');
%ylim([-.1 (thresh + .1)])
title('Ski Drop Eigenvalues','Fontsize',16);
ylabel('Magnitude','Fontsize',14);
xlabel('Index','Fontsize',14);
datalow_inter = Phi(:,I)*modes(I,:);
datasparse_inter = data - abs(datalow_inter);
rcheck = datasparse_inter < 0;
R = datasparse_inter.*rcheck;
datalow = abs(datalow_inter);
datasparse = datasparse_inter + abs(min(datasparse_inter,[],'all'));
figure(101)
i = 1;
subplot(3,1,i)
imshow(reshape(data(:,i),j,k));
subplot(3,1,i+1)
imshow(reshape(datalow(:,i),j,k));
subplot(3,1,i+2)
imshow(reshape(datasparse(:,i),j,k));
sgtitle('Ski Drop Video Decomposed','Fontsize',16)
%% Functions
% Internal Function 
function [data, m, n] = parametrize(predata)
    shapes = size(predata);
    m = shapes(1); n = shapes(2);
    num_frames = shapes(4);
    data = zeros(m*n,num_frames);
    for f = 1:num_frames
        data(:,f) = reshape(rgb2gray(predata(:,:,:,f)),...
            m*n,1);
    end
    data = data/255;
end
function [U,Vs,l] = operator(X1,X2,num_components)
    [U,s,V] = svd(X1,'econ');
    U = U(:,1:num_components);
    s = s(1:num_components,1:num_components);
    V = V(:,1:num_components);
    S = U'*X2*V*diag(1./diag(s));
    [Vs,l] = eig(S);
end