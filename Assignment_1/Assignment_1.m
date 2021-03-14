%% Assignment 1
% William Ojemann
% AMATH 482
% January 10th, 2021
% Submitted - January 27th, 2021
close all; clear; clc;
%% Data Formatting - Provided Code
load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata 5
L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);
% Creating 3D coordinate grids for space and frequency
[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);
% Plotting raw data
figure
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    M = max(abs(Un),[],'all');
    isosurface(X,Y,Z,abs(Un)/M,0.7);
    axis([-10 10 -10 10 -10 10]), grid on
    drawnow;
    pause(0.01);
end
% Plot formatting
xlabel('X direction', 'FontSize', 14)
ylabel('Y direction', 'FontSize', 14)
zlabel('Z direction', 'FontSize', 14)
title('Submarine Recording with Noise', 'FontSize', 18)
%% Identifying Frequency Signature
avg_freqs = zeros(n,n,n); % initialize the 3D matrix
for j = 1:49
   avg_freqs = avg_freqs + fftn(reshape(subdata(:,j),n,n,n));
end
avg_freqs = abs(avg_freqs)/49; % creating the matrix of average frequencies

[maxE, maxIdx] = max((avg_freqs),[],'all','linear'); % finding index of max frequency
[xi, yi, zi] = ind2sub([n, n, n], maxIdx); % converting index to 3D indices
% Converting matrix indices to frequency values
Kxi = Kx(xi,yi,zi);
Kyi = Ky(xi,yi,zi);
Kzi = Kz(xi,yi,zi);
center_freq = [Kxi,Kyi,Kzi];
%% Filtering
% Creating the Gaussian filter
tau = 1;
filterx = exp(-tau.*(Kx - Kxi).^2);
filtery = exp(-tau.*(Ky - Kyi).^2);
filterz = exp(-tau.*(Kz - Kzi).^2);
filter = filterx.*filtery.*filterz;
% Filtering one time stamp for visualization
j = 1;
Un(:,:,:)=reshape(subdata(:,j),n,n,n);
Ut = fftn(Un);
Uft = fftn(Un).*filter;
% Plotting filtering process
figure
view(3)
camlight
lighting gouraud
hold on
subplot(2,2,2)
isosurface(Kx,Ky,Kz,filter,.7);
xlabel('Kx direction', 'FontSize', 12)
ylabel('Ky direction', 'FontSize', 12)
zlabel('Kz direction', 'FontSize', 12)
title('3D Gaussian Filter', 'FontSize', 14)
axis([-10 10 -10 10 -10 10]), grid on
subplot(2,1,2)
isosurface(Kx,Ky,Kz,real(Uft),.7);
xlabel('Kx direction', 'FontSize', 12)
ylabel('Ky direction', 'FontSize', 12)
zlabel('Kz direction', 'FontSize', 12)
title('Filtered Acoustic Emissions', 'FontSize', 14)
axis([-10 10 -10 10 -10 10]), grid on
subplot(2,2,1)
isosurface(Kx,Ky,Kz,real(Ut),.7);
xlabel('Kx direction', 'FontSize', 12)
ylabel('Ky direction', 'FontSize', 12)
zlabel('Kz direction', 'FontSize', 12)
title('Un-Filtered Signal in Freq.', 'FontSize', 14)
axis([-10 10 -10 10 -10 10]), grid on
hold off
%% Finding The Maximum in Space
maxes = zeros(49,3);
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Uft = fftn(Un).*filter; % frequency and shifted and filtered
    Unf = ifftn(Uft); % shifting back to space
    [~,temp_max_idx] = max(abs(Unf),[],'all','linear'); %index of maximum frequency at time stamp
    [x, y, z] = ind2sub([n,n,n], temp_max_idx); %transforming to matrix
    maxes(j,1:3) = [X(x,y,z), Y(x,y,z), Z(x,y,z)]; % storing
end
% Plotting
figure
hold on
g = plot3(maxes(1:end,1),maxes(1:end,2),maxes(1:end,3),'-bo');
p = plot3(maxes(end,1),maxes(end,2),maxes(end,3),'ok');
set(p, 'markerfacecolor', 'g');
set(g, 'MarkerSize',4);
set(p, 'MarkerSize',8);
axis([-10 10 -10 10 -10 10]), grid on
xlabel('X direction', 'FontSize', 14)
ylabel('Y direction', 'FontSize', 14)
zlabel('Z direction', 'FontSize', 14)
title('Submarine Position', 'FontSize', 18)
hold off
%% Poseidon Sub Tracking
% Creating coordinates table
Xcoordinates = maxes(:,1);
Ycoordinates = maxes(:,2);
TrackingCoordinates = table(Xcoordinates,Ycoordinates);
% Plotting coordinates on grid
figure
scatter(Xcoordinates,Ycoordinates);
xlabel('X direction', 'FontSize', 14)
ylabel('Y direction', 'FontSize', 14)
title('Submarine Tracking Coordinates', 'FontSize', 18)
axis([-10 10 -10 10]), grid on