% William Ojemann
% AMATH 482
% 16 Feb 2021
% Assignment 3
clear; close all; clc;
%% Data Parameters
camcount = 3;
tests = 4;
[cams, vids] = parametrize(camcount,tests);
%% Data Aquisition
% clc
% xclick_store = cell(tests,camcount);
% yclick_store = cell(tests,camcount);
% thresh = ones(tests,camcount)*220;
% thresh_check = ones(tests,camcount);
% x_coordinates = cell(tests,camcount);
% y_coordinates = cell(tests,camcount);
% for test = 1:tests
% for cam = 2:camcount
%     xs = []; ys = [];
%     load(cams{test,cam});
%     vid_iter_temp = eval(vids{test,cam});
%     numFrames = size(vid_iter_temp,4);
%     clear vid_iter_gray;
%     for k = 1:numFrames
%         if cam == 3
%             vid_iter_gray(:,:,k) = rgb2gray(vid_iter_temp(:,:,:,k))';
%         else
%             vid_iter_gray(:,:,k) = rgb2gray(vid_iter_temp(:,:,:,k));
%         end
%     end
%     [offx,boundx,offy,boundy] = loc(vid_iter_gray,cams,test,cam,100);
%     xclick_store{test,cam} = [offx boundx];
%     yclick_store{test,cam} = [offy boundy];
%     
%     check = 0;
%     while ~ischar(check) && thresh_check(test,cam)
%         for i = 1:4:numFrames
%             X_window = vid_iter_gray(offy:boundy,offx:boundx,i);
%             figure(101)
%             imshow(X_window>=thresh(test,cam));drawnow
%         end
%         disp(['Threshold: ' num2str(thresh(test,cam))]);
%         check = input('Give thresh. diff. (e.g. 10,-5), "reloc" to change bounds, and y when done: ','s');
%         if check == "reloc"
%             [offx,boundx,offy,boundy] = loc(vid_iter_gray,cams,test,cam,100);
%             check = 0;
%             continue
%         end
%         check = str2double(check); % returns Nan if not a number char
%         if ~isnan(check) % checks for number
%             thresh(test,cam) = thresh(test,cam) + check; %adds number to thresh
%         else
%             check = 'done'; % ends loop if non number was input
%         end
%     end
% 
%     for j = 1:numFrames
%         X_window = vid_iter_gray(offy:boundy,offx:boundx,j);
%         X_binary = X_window >= thresh(test,cam);
%         if sum(sum(X_binary))
%             [y,x] = find(X_binary); %(:,200:end));
%             x = round(mean(x));
%             y = round(mean(y));
%             Y = zeros(size(X_window)); %(:,200:end)));
%             Y(x,y) = 1;
%             xs = [xs x+offx];
%             ys = [ys y+offy];
%         else
%             xs = [xs xs(end)];
%             ys = [ys ys(end)];
%         end
%     end
%     x_coordinates{test,cam} = xs;
%     y_coordinates{test,cam} = ys;
% end
% end
% for i = 1:tests
%     figure(i)
%     subplot(2,1,1)
%     plot(x_coordinates{i,1})
%     subplot(2,1,2)
%     plot(y_coordinates{i,1}) 
% end

%% File Storage
% % clicks.x = xclick_store;
% % clicks.y = yclick_store;
% % save('clicks.mat', '-struct', 'clicks');
% % coordinates.x = x_coordinates;
% % coordinates.y = y_coordinates;
% % save('coordinates.mat', '-struct', 'coordinates');
% % save('thresholds.mat','thresh');
%% Load Files
[coordinates, thresholds, click_coordinates] = load_data();
x_coordinates = coordinates.x;
y_coordinates = coordinates.y;
%% Data formatting
[data,M,~,~] = resize_data(x_coordinates,y_coordinates,tests,camcount);
%% Principal Component Analysis
for i = 1:tests
    svd_analysis_plotting(data,i,i*10, camcount)
end
%% Analysis of Principal Components
plot_pc_analysis(data);
%% Functions
function svd_analysis_plotting(data,testset,fignum,camcount)
    %%%%% Plotting Reconstructions
    % Perform SVD
    [U,S,V] = svd(data{testset},'econ');
    % Create rank 1 and 2 reconstructions
    low_rank_one = U(:,1)*S(1,1)*V(:,1)';
    low_rank_two = U(:,1:2)*S(1:2,1:2)*V(:,1:2)';
    figure(fignum)
    % Iterate through feature sets and plot approximations along with raw
    % data for each X and Y directions.
    for i = 1:camcount
        l = length(data{testset});
        subplot(1,2,1)
        plot(1:l,data{testset}(2*i-1,:)+2*i,'k-')
        hold on
        plot(1:l,low_rank_one(2*i-1,:)+2*i,'r-.')
        p1 = plot(1:l,low_rank_two(2*i-1,:)+2*i,'b-.');
        p1.Color(4) = 0.5;
        legend('Raw','Rank 1','Rank 2')
        yticks(linspace(1,camcount,camcount)*2)
        yticklabels({'Cam 1', 'Cam 2', 'Cam 3'})
        title('X','Fontsize',16);
        ylabel('Unity Scaled Position','Fontsize',14)
        xlabel('Frame','Fontsize',14)
        subplot(1,2,2)
        plot(1:l,data{testset}(2*i,:)+2*i,'k-')
        hold on
        plot(1:l,low_rank_one(2*i,:)+2*i,'b-.')
        p2 = plot(1:l,low_rank_two(2*i,:)+2*i,'r-.');
        p2.Color(4) = 0.5;
        yticks(linspace(1,camcount,camcount)*2)
        yticklabels({'Cam 1', 'Cam 2', 'Cam 3'})
        title('Y','Fontsize',16);
        ylabel('Unity Scaled Position','Fontsize',14)
        xlabel('Frame','Fontsize',14)
    end
    hold off
    sgtitle(['Test ', num2str(testset), ' Positions'],'Fontsize',18);
    
    %%%%% Plotting Decomposition Properties
    figure(fignum+1)
    % Plot the percent total kinetic energy
    subplot(3,1,1)
    sigmas = diag(S);
    energies = zeros(1,length(sigmas));
    errors = zeros(1,length(sigmas));
    for i = 1:length(sigmas)
        energies(i) = sum(sigmas(i).^2)/sum(sigmas.^2);
        low_rank_recon = U(:,1:i)*S(1:i,1:i)*V(:,1:i)';
        errors(i) = sum((data{testset} - low_rank_recon).^2,'all')./numel(low_rank_recon);
    end
    scatter(1,energies(1),'k');
    hold on
    scatter(2,energies(2),'b')
    scatter(3,energies(3),'r')
    scatter(4:length(sigmas),energies(4:end))
    hold off
    ylim([0 1])
    ylabel('Percent Total Energy','Fontsize',14);
    xlabel('Principal Component','Fontsize',14)
    title('Principal Component Analysis','Fontsize',16)
    xticks(1:length(sigmas))
    % Plot the MSE of reconstruction
    subplot(3,1,3)
    scatter(1,errors(1),'k');
    hold on
    scatter(2,errors(2),'b');
    scatter(3,errors(3),'r');
    scatter(4:length(sigmas),errors(4:end));
    hold off
    ylabel('MSE','Fontsize',14);
    xlabel('Reconstruction Rank','Fontsize',14);
    xticks(1:length(sigmas))
    % Plot the time evolution of each principal component
    subplot(3,1,2)
    plot(1:l,V(:,1)','k',1:l,V(:,2)','b',1:l,V(:,3)','r')
    ylabel('Mode Amplitude','Fontsize',14)
    xlabel('Frame','Fontsize',14)
    legend('1^{st} Mode', '2^{nd} Mode', '3^{rd} Mode')
end

function [unity_data,M,x_coordinates_crop, y_coordinates_crop] = resize_data(x_coordinates,y_coordinates,tests,camcount)
    % Initialize cropped data
    x_coordinates_crop = cell(tests,camcount);
    y_coordinates_crop = cell(tests,camcount);
    data = cell(1,tests);
    longest_film = ones(1,tests)*10000;
    demeaned_data = cell(1,tests);
    unity_data = cell(1,tests);
    
    % Iterate through test
    for j = 1:tests
        % Find the shortest camera frame count for
        % each test
        for i = 1:camcount
            if length(x_coordinates{j,i}) < longest_film(j)
                longest_film(j) = length(x_coordinates{i});
            end
        end
        % Resize the data into the cropped format
        data{j} = nan(2*camcount,longest_film(j));

        for i = 1:camcount
            if i == 2 && j~= 3 && j~= 4
                x_coordinates_crop{j,i} = x_coordinates{j,i}(16:longest_film(j)+15);
                y_coordinates_crop{j,i} = y_coordinates{j,i}(16:longest_film(j)+15);
            elseif j == 4 && i == 2
                x_coordinates_crop{j,i} = x_coordinates{j,i}(11:longest_film(j)+10);
                y_coordinates_crop{j,i} = y_coordinates{j,i}(11:longest_film(j)+10);
            else
            x_coordinates_crop{j,i} = x_coordinates{j,i}(1:longest_film(j));
            y_coordinates_crop{j,i} = y_coordinates{j,i}(1:longest_film(j));
            end
            data{j}(2*i-1:2*i,:) = [x_coordinates_crop{j,i} ; y_coordinates_crop{j,i}];
        end
        % Demean data for PCA and find maximum value for unity scaling
        demeaned_data{j} = data{j} - mean(data{j},2);
        M = max(demeaned_data{j},[],2);
        unity_data{j} = demeaned_data{j} ./ M;
    end
end

function [offx,boundx,offy,boundy] = loc(image,cams,test,cam,fignum)
    figure(fignum)
    for j = 1:4:size(image,3)
        imshow(image(:,:,j)); drawnow
        title(cams{test,cam})
    end
    disp('Pick two points as left & right bounds of movement');
    [xclick,~] = getpts;
    disp('Now pick low and high bounds');
    [~,yclick] = getpts;
    offx = round(xclick(1));
    boundx = round(xclick(2));
    offy = round(yclick(2));
    boundy = round(yclick(1));
end

function [coordinates, thresholds, click_coordinates] = load_data()
    % Internal Function
    % Load previously acquired data
    coordinates = load('coordinates.mat');
    thresholds = load('coordinates.mat');
    click_coordinates = load('coordinates.mat');
end

function [cams, vids] = parametrize(camcount,tests)
    % Internal Function
    % Initialize variables for loading data
    cams = cell(tests,camcount);
    vids = cell(tests,camcount);
    for cam = 1:camcount
        for test = 1:tests
            cams{test,cam} = ['cam' num2str(cam) '_' num2str(test) '.mat'];
            vids{test,cam} = ['vidFrames' num2str(cam) '_' num2str(test)];
        end
    end
end

function plot_pc_analysis(data)
figure
%for i = 1:4
    [~,~,v] = svd(data{3},'econ');
    %subplot(2,2,i)
    n = length(v);
    k = 2*pi/n*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);
    f1 = abs(fftshift(fft(v(:,1))));
    f2 = abs(fftshift(fft(v(:,2))));
    [fmax,fmaxidx] = max(f1);
    [fmax2,fmaxidx2] = max(f2);
    plot(ks,f1/fmax);
    hold on
    plot(ks,f2/fmax2);
    title(['Test ',num2str(3),' PC Frequency Spectrum'],'Fontsize',16);
    ylabel('Scaled Amplitude','Interpreter','latex','Fontsize',14)
    xlabel('Frequency ($\frac{rad}{frame}$)','Interpreter','latex','Fontsize',14)
    legend('PC 1', 'PC 2');
    disp(num2str(abs(ks(fmaxidx))));
    disp(num2str(abs(ks(fmaxidx2))));
%end
end
% https://pungenerator.org/puns?q=principal