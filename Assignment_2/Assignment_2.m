% Assignment 2
% William Ojemann
% AMATH 482
% 1/31/2021
clear; close all; clc;
%%
% Audio Clip Plotting
[gnr, Fsgnr] = audioread('GNR.m4a');
[floyd, Fsfloyd] = audioread('Floyd.m4a');
figure(1)
set(gca,'Fontsize',14)
subplot(2,1,1)
plot((1:length(gnr))/Fsgnr,gnr);
xlabel('Time [sec]','Fontsize',14); ylabel('Amplitude','Fontsize',14);
title("Sweet Child o' Mine",'Fontsize',16);
subplot(2,1,2)
plot((1:length(floyd))/Fsfloyd,floyd);
xlabel('Time [sec]','Fontsize',14); ylabel('Amplitude','Fontsize',14);
title('Comfortably Numb','FontSize',16);
%% Guns 'n Roses
clc
[gnr, Fsgnr] = audioread('GNR.m4a');
p8 = audioplayer(gnr,Fsgnr);
playblocking(p8);

[t, ks, ~, ~, L] = formatting(gnr, Fsgnr);
a = 10;
n_taus = 101;
tauset = 0:L/(n_taus-1):L;
fignum = 2;
gabor_process(t,ks,gnr,tauset,a,fignum)

fignum = 3;
ylimits = [0 800];
[Y_gnr,filter_indices] = spec_plot(t,ks,gnr,tauset,a,fignum, ylimits);
set(gca,'Fontsize',14)
title("Spectrogram for Sweet Child o' mine Solo",'Fontsize', 16)
ylabel('Frequency (Hz)','Fontsize', 14)
xlabel('Time (s)','Fontsize', 14)
[~,score] = max(Y_gnr,[],1);
ksf = ks(filter_indices);
scoregnr = ksf(score);
score_plot(t,sort([277.18,554.37,415.3,369.99,739.99,698,311.13]),{});
figure(100)
hold on
scatter(tauset,scoregnr,'r')
set(gca,'Fontsize',14)
ylabel('Notes','Fontsize', 14)
xlabel('Time (s)','Fontsize', 14)
title("Score for Sweet Child o' mine Solo",'Fontsize', 16)
score_plot(t,sort([277.18,554.37,415.3,369.99,739.99,698,311.13]),{'Db','Eb','Gb','Ab','Db','F','Gb'});
%% Hyper Parameter Tuning
[gnr, Fsgnr] = audioread('GNR.m4a');
[t, ks, ~, ~, L] = formatting(gnr, Fsgnr);
a = [.1, 10, 1000, 100000];
n_taus = 101;
tauset = 0:L/(n_taus-1):L;
fignum = 106;
ylimits = [0 800];
for j = 1:length(a)
    subplot(4,1,j)
    spec_plot(t,ks,gnr,tauset,a(j),fignum, ylimits);
    xlabel('Time (s)', 'Fontsize', 14)
    ylabel('Frequency (Hz)', 'Fontsize', 14)
    title(['Spectrogram a = ' num2str(a(j))]);
end
%% Pink Floyd
% %clear; close all; clc;
[floyd, Fsfloyd] = audioread('Floyd.m4a');
p8 = audioplayer(floyd,Fsfloyd);
playblocking(p8);
%% Baseline Isolation
clc
[floyd, Fsfloyd] = audioread('Floyd.m4a');
a = 15;
n_taus = 251;
[t, ks, floyd, n, L] = formatting(floyd, Fsfloyd);
ylimits = [50 250];
fignum = 4;
tauset = 0:L/(n_taus-1):L;
[Y_floyd,filter_indices] = spec_plot(t,ks,floyd,tauset,a,fignum, ylimits);
set(gca,'Fontsize',14)
ylabel('Notes','Fontsize', 14)
xlabel('Time (s)','Fontsize', 14)
title("Spectrogram for Comfortably Numb Bass",'Fontsize', 16)
[~,score] = max(Y_floyd,[],1);
ksf = ks(filter_indices);
scorefloyd = ksf(score);
filter_width = 4;
hold on
[filter_spec] = isolation(ksf,Y_floyd,scorefloyd,tauset,filter_width,5);
score_plot(t,[82.407,92.499,97.999,110,123.47,146.83,164.81,185,196,246.94],{});
set(gca,'Fontsize',14)
ylabel('Notes','Fontsize', 14)
xlabel('Time (s)','Fontsize', 14)
title("Iso. Spec. for Comfortably Numb Bass",'Fontsize', 16)

[~,filter_score] = max(filter_spec,[],1);
filter_score = ksf(filter_score);
figure(101)
hold on
scatter(tauset,filter_score,'r')
score_plot(t,[82.407,92.499,97.999,110,123.47,146.83,164.81,185,196,246.94]...
    ,{'E','F#','G','A','B','D','E','F#','G','B'});
set(gca,'Fontsize',14)
ylabel('Notes','Fontsize', 14)
xlabel('Time (s)','Fontsize', 14)
title("Score for Comfortably Numb Bass",'Fontsize', 16)
%% Guitar Solo Identification
clc
[floyd, Fsfloyd] = audioread('Floyd.m4a');
l = length(floyd);
a = 100;
ylimits = [300 1000];
n_taus = 701;
[~, ~, floyd, ~, ~] = formatting(floyd, Fsfloyd);
figs = 10;
score_set = nan(figs,n_taus);
figure(7)
%subplot(figs/2,2,1)
[t, ks, floyds, ~, L] = formatting(floyd(1:floor(l/figs)), Fsfloyd);
fignum = 7;
tauset = 0:L/(n_taus-1):L;
[Y_floyd,filter_indices] = spec_plot(t,ks,floyds,tauset,a,[], ylimits);
[~,score] = max(Y_floyd,[],1);
ksf = ks(filter_indices);
score_set(1,:) = ksf(score);
filter_width = 5;
isolation(ksf,Y_floyd,score_set(1,:),tauset,filter_width,fignum);
score_plot(t,[329.63,369.99,440,493.88,554.37,587.33,622.25,659.26],{});
set(gca,'Fontsize',16, 'YScale', 'log')
ylabel('Frequency (Hz)','Fontsize', 14)
xlabel('Time (t)','Fontsize', 14)
title("Iso. Spec. for Comfortably Numb Solo (0-6s)",'Fontsize', 16)

figure(104)
hold on
set(gca,'Fontsize',16, 'YScale', 'log')
scatter(tauset,score_set(1,:),'r')
score_plot(t,[329.63,369.99,440,493.88,554.37,587.33,622.25,659.26],...
    {'E','F#','A','B','C#','D','D#','E'});
set(gca,'Fontsize',14)
ylabel('Notes','Fontsize', 14)
xlabel('Time (s)','Fontsize', 14)
title("Score for Comfortably Solo (0-6s)",'Fontsize', 16)
for j = 2:figs
    %fignum = 6+j;
    subplot(figs/2,2,j)
    [t, ks, floyds,~,~] = formatting(...
        floyd(floor(l/figs*(j-1)):floor(l/figs*j)-1), Fsfloyd);
    [Y_floyd,filter_indices] = spec_plot(t,ks,floyds,tauset,a,fignum, ylimits);
    [~,score] = max(Y_floyd,[],1);
    ksf = ks(filter_indices);
    score_set(j,:) = ksf(score);
    isolation(ksf,Y_floyd,score_set(j,:),tauset,filter_width,fignum);
    set(gca,'Fontsize',16, 'YScale', 'log')
    ylabel('Frequency (Hz)','Fontsize', 14)
    xlabel('Time (t)','Fontsize', 14)
    score_plot(t,5*[82.407,92.499,97.999,110,123.47,164.81,185,196],{})
end
%% Plotting Guitar Solo Notes
ltauset = 0:l/Fs/(n_taus*figs-1):l/Fs;
linscore = [];
for j =  1:figs
    linscore = [linscore score_set(j,:)];
end

figure(102)
scatter(ltauset,linscore)
set(gca,'Fontsize',16, 'YScale', 'log')
ylabel('Frequency (Hz)','Fontsize', 14)
xlabel('Time (t)','Fontsize', 14)
Title('Comfortably Numb Guitar Solo')
%% Functions
% Initial Data Formatting -- internal function
function [t, ks, y, n, L] = formatting(y, Fs)
    if mod(length(y),2) == 1
        y = y(1:end-1);
    end
    % Feature Extraction
    L = length(y)/Fs; % record time in seconds
    n = length(y);
    t2 = linspace(0,L,n+1); t = t2(1:n);
    k = (1/L)*[0:(n)/2-1 -(n)/2:-1]; ks = fftshift(k); 
end

% Plotting Filtering Process -- internal function
function gabor_process(t,ks,y,tauset,a,fignum)
    if ~isempty(fignum)
        figure(fignum);
    end
    for tau = tauset
        giter = exp(-a*(t-tau).^2);
        subplot(3,1,1)
        hold on
        plot(t,y)
        plot(t,giter);
        xlabel('Time (s)', 'FontSize', 14)
        title('Gabor Transform', 'FontSize', 16);
        yf = giter'.*y;
        subplot(3,1,2)
        plot(t,yf)
        yft = fft(yf);
        xlabel('Time (s)', 'FontSize', 14)
        subplot(3,1,3)
        plot(ks,fftshift(abs(yft)))
        xlabel('Frequency (Hz)', 'FontSize', 14)
        drawnow
        pause(0.1)
        clf
    end
end

% Plotting Spectrogram -- internal function
function [y_spec,filter_indices] = spec_plot(t,ks,y,tauset,a,fignum, ylimits)
    filter_indices = (ks >= ylimits(1) & ks <= ylimits(2));
    y_spec = nan(sum(filter_indices),length(tauset));
    for j = 1:length(tauset)
        giter = exp(-a*(t-tauset(j)).^2);
        yf = y.*giter';
        yft = fft(yf);
        yft = fftshift(abs(yft));
        yf = yft(filter_indices);
        y_spec(:,j) = yf;
    end
    if fignum
        figure(fignum)
        pcolor(tauset,ks(filter_indices),y_spec);
        set(gca,'ylim',ylimits,'Fontsize',16)
        shading interp
        colormap(hot)
        colorbar
    end
end

% Note Label Plotting -- internal function
function music = score_plot(t,notes,notelabels)
    hold on
    music = nan(length(notes),2);
    tset = [t(1) t(end)];
    for note = notes
       plot(tset, [note note], 'w-.','LineWidth',1);
    end
    yticks(notes);
    if ~isempty(notelabels)
        yticklabels(notelabels);
    end
    hold off
end

% Musical Part Isolation -- internal function
function [filter_spec] = isolation(ks,spec,score,tauset,filter_width,fignum)
    filter_spec = nan(length(ks),length(tauset));
    for j = 1:length(tauset)
        ygt = spec(:,j);
        note = score(j);
        filter_indices = ks < (note+filter_width/2) & ks > (note-filter_width/2);
        ygft = ygt'.*filter_indices;
        filter_spec(:,j) = ygft;
    end
    figure(fignum)
    pcolor(tauset,ks,log(filter_spec+1));
    set(gca,'Fontsize',16, 'YScale', 'log')
    shading interp
    colormap(hot)
    colorbar
end