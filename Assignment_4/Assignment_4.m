% William Ojemann
% AMATH 482
% Assignment 4
% 27 FEB 2021
clear; close all; clc;
%% Data Loading and Processing
[images_train, labels_train] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[images_test, labels_test] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
[num_indices_train,features_train] = initial_processing(images_train,labels_train);
[num_indices_test,features_test] = initial_processing(images_test,labels_test);
%% SVD
[Utr,Str,Vtr] = svd(features_test,'econ');
m = size(images_test,1);
n = size(images_test,2);
figure
subplot(1,2,2)
scatter(1:length(diag(Str)),diag(Str))
set(gca,'yscale','log')
ylabel('Magnitude','Fontsize',14)
xlabel('Singular Value','Fontsize',14)
subplot(1,2,1)
scatter(1:length(diag(Str)),diag(Str))
ylabel('Magnitude','Fontsize',14)
xlabel('Singular Value','Fontsize',14)
sgtitle('Training Data Singular Values','Fontsize',16)
figure
for k = 1:9
    subplot(3,3,k)
    X = reshape(Utr(:,k),m,n);
    imshow(rescale(X))
    title(['PC ', num2str(k)],'Fontsize',14)
end
sgtitle('Training Data Principal Components','Fontsize',16)
%% Projection Color Scheme
colors = [0 0 0];
grad = linspace(0.2,1,3)';
gradalt = linspace(1,.2,3)';
temp = [0; 0; 0];
onethree = [gradalt temp grad];
foursix = [grad gradalt temp];
sevnine = [temp grad gradalt];
colors = [colors;
          onethree;
          foursix;
          sevnine];
colormaps = zeros(length(labels_test),3);
for i = 1:length(labels_test)
    label = labels_test(i);
    colormaps(i,:) = colors(label+1,:);
end
%% PC Projection Plotting
cols = [2,3,5];
scaled_modes = (Str*Vtr')';
projs = scaled_modes(:,cols);
figure
scatter3(projs(:,1),projs(:,2),projs(:,3),[],colormaps)
xlabel(['Principal Component ',num2str(cols(1))],'Fontsize',14)
ylabel(['Principal Component ',num2str(cols(2))],'Fontsize',14)
zlabel(['Principal Component ',num2str(cols(3))],'Fontsize',14)
title(['Projection Onto PC ',num2str(cols(1)),', ',num2str(cols(2)),...
    ', and ',num2str(cols(3))],'Fontsize',16)
%% Wavelet and SVD
% % X = num_wave(images_train);
% load('images_train_wave.mat')
% [mw,nw,~] = size(X);
% features_w = double(reshape(X,mw*nw,size(X,3)));
% [uw,sw,vw] = svd(features_w,'econ');
%% Wavelet and SVD plotting
% figure
% for k = 1:9
%     subplot(3,3,k)
%     X = reshape(uw(:,k),mw,nw);
%     imshow(rescale(X))
% end
%% One Sample Analysis
figure
subplot(3,3,1:3)
imshow(images_train(:,:,1))
for i = 1:3
    subplot(3,3,i+3)
    X = reshape(Utr(:,i),m,n);
    imshow(rescale(X))
    title(['PC ', num2str(i)],'Fontsize',14)
end
subplot(3,3,7:9)
Y = Vtr(1,1:3);
plot(Y,'-o')
title('Principal Component Weights','Fontsize',14)
sgtitle('SVD Decomposition','Fontsize',16)
%% Rank Evaluation
nums = [0,7];
ranks = [1 2 3 5 7 10 15 20 25 30 40 50 70 100 200 300 400 600 784];
%ranks = [2];
score = zeros(1,length(ranks));
features = features_train;
num_indices = num_indices_train;
for r = 1:length(ranks)
    num_components = ranks(r);
[alpha,beta,w,threshold,U] = two_feature_lda(nums(1),nums(2),features,...
                        num_indices,num_components);
    [labels,labels_pred] = lda_predict(nums(1),nums(2),num_indices_test,...
                            features_test,U,w,threshold);
    con_mat_test = make_confusion_matrix(labels,labels_pred,0);
    score(r) = mean(diag(con_mat_test));
end
%% Rank Evaluation Plotting
nums = [0,7];
ranks = [1 2 3 5 7 10 15 20 25 30 40 50 70 100 200 300 400 600 784];
load('rank_analysis.mat')
figure
semilogx(ranks,score)
xlabel('Log of SVD Modes Used for Prediction','Fontsize', 14)
ylabel('Probablility of Correct Prediction','Fontsize',14)
title(['LDA Predictions Classifying between ',num2str(nums(1)),' and ',num2str(nums(2))],'Fontsize',16)
%% Two Features - Prep
num1 = 0;
num2 = 7;
num_components = 10;
[vone,vtwo,w,threshold,U] = two_feature_lda(num1,num2,features_train,num_indices_train,num_components);
figure(100)
subplot(1,2,1)
plot(vone,zeros(length(vone)),'ob')
hold on
plot(vtwo,ones(length(vtwo)),'or')
title(['Zero: ', num2str(num1), ' One: ', num2str(num2)])
set(gca,'Fontsize',14)
ylim([0 1.7]);
%%
subplot(1,2,2)
histogram(vone);
hold on, plot([threshold threshold], [0 650],'r')
set(gca,'Ylim',[0 625],'Fontsize',14)
subplot(1,2,2)
histogram(vtwo); %hold on, plot([threshold threshold], [0 650],'r')
title(['Distribution of ' num2str(num1) ' and ' num2str(num2)])
set(gca,'Ylim',[0 625],'Fontsize',14)
%% Confusion Matrices
% Confusion Matrix - Test
[labels,labels_pred] = lda_predict(num1,num2,num_indices_test,features_test,U,w,threshold);
con_mat_test = make_confusion_matrix(labels,labels_pred);
figure
subplot(1,2,1)
cm = confusionchart(round(con_mat_test*100),[num1 num2]);
cm.Title = ['Correct Classification - Test'];
cm.Normalization = 'row-normalized';
set(gca,'Fontsize',14)
% Confusion Matrix - Train
[labels,labels_pred] = lda_predict(num1,num2,num_indices_train,features_train,U,w,threshold);
con_mat_test = make_confusion_matrix(labels,labels_pred);
subplot(1,2,2)
cm = confusionchart(round(con_mat_test*100),[num1 num2]);
cm.Title = ['Correct Classification - Train'];
cm.Normalization = 'row-normalized';
set(gca,'Fontsize',14)
%% Digit Classification
% digits = 3:9;
% score = nan(length(digits));
% for digit = digits
%     for digit2 = digits
%         if digit == digit2
%             continue
%         end
%         num1 = digit;
%         num2 = digit2;
%         num_components = 10;
%         [vone,vtwo,w,threshold,U] = two_feature_lda(num1,num2,features_train,num_indices_train,num_components);
%         [labels,labels_pred] = lda_predict(num1,num2,num_indices_test,features_test,U,w,threshold);
%         con_mat_test = make_confusion_matrix(labels,labels_pred);
%         if digit == 3 && digit2 == 5
%             figure
%             confusionchart(round(con_mat_test*100))
%         end
%         score(digit+1,digit2+1) = mean(diag(con_mat_test));
%         disp([num1 num2 score(digit+1,digit2+1)])
%     end
% end
%%
clear score;
load 'two_digit_scores.mat'
score = [score; nan(1,length(score))];
score = [score nan(length(score),1)];
figure
pcolor(0:10,0:10,score)
colorbar
title('Probability of Correct Classification','Fontsize',16)
xlabel('Number 1','Fontsize', 14)
ylabel('Number 2','Fontsize',14)
[maxprob,maxidx] = max(score,[],'all','linear');
[X,Y] = ind2sub(size(score),maxidx);
%% MATLAB Multivariate Classification
num1 = 2; num2 = 1; num3 = 5;
nums = sort([num1 num2 num3]);
features = features_train;
num_indices = num_indices_train;
num_components = 20;
[values, w, ~, U] = mvlda(nums,features,...
                        num_indices,num_components);
meds = zeros(1,length(values));
for i = 1:length(values)
    meds(i) = median(values{i});
end
threshold = [(meds(1)+meds(2))/2,(meds(2)+meds(3))/2];
figure(101)
subplot(1,3,1)
histogram(values{1}); 
hold on, plot([threshold(1) threshold(1)], [0 800],'r')
plot([threshold(2) threshold(2)], [0 800],'r')
set(gca,'Fontsize',14);
title(num2str(nums(1)))
subplot(1,3,2)
histogram(values{2}); hold on, plot([threshold(1) threshold(1)], [0 800],'r')
plot([threshold(2) threshold(2)], [0 800],'r')
set(gca,'Fontsize',14);
title(num2str(nums(2)))
subplot(1,3,3)
histogram(values{3}); hold on, plot([threshold(1) threshold(1)], [0 800],'r')
plot([threshold(2) threshold(2)], [0 800],'r')
set(gca,'Fontsize',14);
title(num2str(nums(3)))
sgtitle('Three Number LDA Classification Distributions','Fontsize',16)
%% SVM
% % model = fitcecoc(features_train',labels_train);
load('svm_model.mat')
labels_pred = predict(model,features_test');
con_mat_test = make_confusion_matrix(labels_test,labels_pred,1);
figure
subplot(1,2,1)
cm = confusionchart(round(con_mat_test*100),0:9);
cm.Title = 'Probability of Correct Classification - SVM';
cm.Normalization = 'row-normalized';
set(gca,'Fontsize',14)
disp(['Average Score: ' num2str(mean(diag(con_mat_test)))])
%% Binary Tree Optimization
% clc
% splits = round(logspace(0,3.5,10));
% classErrors = zeros(2,length(splits));
% for i = 1:length(splits)
%     treecval=fitctree([features_train features_test]',[labels_train; labels_test],'MaxNumSplits',splits(i),'Crossval','on');
%     tree = fitctree([features_train]',[labels_train],'MaxNumSplits',splits(i));
%     %view(tree.Trained{1},'Mode','graph');
%     labels_pred_iter = predict(tree,features_test');
%     con_mat_iter = make_confusion_matrix(labels_test,labels_pred_iter,1);
%     classErrors(1,i) = kfoldLoss(treecval);
%     classErrors(2,i) = mean(diag(con_mat_iter));
%     disp(i);
% end
%% Optimization Plotting
load('binary_tree_results.mat')
splits = round(logspace(0,3.5,10));
figure
plot(splits,classErrors)
title('Evaluation of Binary Tree Classification','Fontsize',16)
legend('Kfold Loss','Prediciton Accuracy')
xlabel('Max Tree Splits','Fontsize',14)
ylabel('Correct Prediction Probability/Loss','Fontsize',14)
%% Optimized tree
tree = fitctree([features_train]',[labels_train],'MaxNumSplits',1300);
labels_pred_iter = predict(tree,features_test');
con_mat_iter = make_confusion_matrix(labels_test,labels_pred_iter,1);
subplot(1,2,2)
cm = confusionchart(round(con_mat_iter*100),0:9);
cm.Title = 'Probability of Correct Classification - Tree';
cm.Normalization = 'row-normalized';
set(gca,'Fontsize',14)
%% Best classification
num1 = 0;
num2 = 1;
num_components = 40;
[~,~,w,threshold,U] = two_feature_lda(num1,num2,features_train,num_indices_train,num_components);
[labels,labels_pred] = lda_predict(num1,num2,num_indices_test,features_test,U,w,threshold);
con_mat_test_lda = make_confusion_matrix(labels,labels_pred);
subplot(1,3,1)
cm = confusionchart(round(con_mat_test_lda*100),[0 1]);
cm.Title = 'Probability of Correct Classification - LDA';
cm.Normalization = 'row-normalized';
set(gca,'Fontsize',14)
disp('Performing SVM')
num1 = num1+1;
num2 = num2+1;
trainidxs = [num_indices_train{num1}; num_indices_train{num2}];
testidxs = [num_indices_test{num1}; num_indices_test{num2}];
features_train_1 = features_train(:,trainidxs); labels_train_1 = labels_train(trainidxs);
features_test_1 = features_test(:,testidxs); labels_test_1 = labels_test(testidxs);
model = fitcecoc(features_train_1',labels_train_1);
labels_pred = predict(model,features_test_1');
con_mat_test_svm = make_confusion_matrix(labels_test_1,labels_pred,1);
subplot(1,3,2)
cm = confusionchart(round(con_mat_test_svm*100),0:1);
cm.Title = 'Probability of Correct Classification - SVM';
cm.Normalization = 'row-normalized';
set(gca,'Fontsize',14)
disp('Performing Binary Tree')
tree = fitctree([features_train_1]',[labels_train_1],'MaxNumSplits',1300);
labels_pred_iter = predict(tree,features_test_1');
con_mat_iter = make_confusion_matrix(labels_test_1,labels_pred_iter,1);
subplot(1,3,3)
cm = confusionchart(round(con_mat_iter*100),0:1);
cm.Title = 'Probability of Correct Classification - Tree';
sgtitle('Confusion Matrices for Best Classification')
cm.Normalization = 'row-normalized';
set(gca,'Fontsize',14)
%% Worst Classification
num1 = 3;
num2 = 5;
num_components = 40;
[vone,vtwo,w,threshold,U] = two_feature_lda(num1,num2,features_train,num_indices_train,num_components);
[labels,labels_pred] = lda_predict(num1,num2,num_indices_test,features_test,U,w,threshold);
con_mat_test_lda = make_confusion_matrix(labels,labels_pred);
subplot(1,3,1)
cm = confusionchart(round(con_mat_test_lda'*100),[num1 num2]);
cm.Title = 'Probability of Correct Classification - LDA';
cm.Normalization = 'row-normalized';
set(gca,'Fontsize',14)
disp('Performing SVM')
num1 = num1+1;
num2 = num2+1;
trainidxs = [num_indices_train{num1}; num_indices_train{num2}];
testidxs = [num_indices_test{num1}; num_indices_test{num2}];
features_train_1 = features_train(:,trainidxs); labels_train_1 = labels_train(trainidxs);
features_test_1 = features_test(:,testidxs); labels_test_1 = labels_test(testidxs);
% model = fitcecoc(features_train_1',labels_train_1);
load('svm_model_twofeature.mat')
labels_pred = predict(model,features_test_1');
% con_mat_test_svm = make_confusion_matrix(labels_test_1,labels_pred);
subplot(1,3,2)
cm = confusionchart(labels_test_1,labels_pred);%round(con_mat_test_svm*100),0:1);
cm.Title = 'Correct Classification - SVM';
cm.Normalization = 'row-normalized';
set(gca,'Fontsize',14)
disp('Performing Binary Tree')
tree = fitctree([features_train_1]',[labels_train_1],'MaxNumSplits',1300);
labels_pred_iter = predict(tree,features_test_1');
%con_mat_iter = make_confusion_matrix(labels_test_1,labels_pred_iter);
subplot(1,3,3)
cm = confusionchart(labels_test_1,labels_pred_iter);%round(con_mat_iter*100),0:1);
cm.Title = ['Correct Classification - Tree - Avg: '];
cm.Normalization = 'row-normalized';
set(gca,'Fontsize',14)
%% Functions

% Multivariate Linear Discriminate Analysis
function [values, w, threshold, U] = mvlda(nums,features,...
                        num_indices,num_components)
    
    l = length(nums);
    values = cell(1,l);
    featuresets = [];
    featuresets_svd = cell(1,l);
    lengths = zeros(1,l+1);
    for i = 1:l
        num = nums(i) + 1;
        featuresets = [featuresets features(:,num_indices{num})];
        lengths(i+1) = length(features(:,num_indices{num}));
    end
    [U,S,V] = svd(featuresets,'econ');
    U = U(:,1:num_components);
    
    featuressvd = S(1:num_components,1:num_components)*V(:,1:num_components)';
    means = nan(size(featuressvd,1),l);
    for i = 1:l
        featuresets_svd{i} = featuressvd(:,lengths(i)+1:lengths(i)+lengths(i+1));
        means(:,i) = mean(featuressvd(:,lengths(i)+1:lengths(i)+lengths(i+1)),2);
    end
    
    meantot = mean(means,2);
    Sw = 0;
    for i = 1:l
        for j = 1:lengths(i+1)
            Sw = Sw + (featuresets_svd{i}(:,j)-means(:,i))*(featuresets_svd{i}(:,j)-means(:,i))';
        end
    end
    Sb = 0;
    for k = 1:l
        Sb = Sb + (means(:,k) - meantot)*(means(:,k) - meantot)';
    end
    [V2,D] = eig(Sb,Sw);
    [~, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    for i = 1:l
        values{i} = w'*featuresets_svd{i};
    end
    if l == 2
        i1 = length(values{1});
        i2 = 1;
        while values{1}(i1) > values{2}(i2)
           i1 = i1-1; i2 = i2 + 1;
        end
        threshold = (values{1}(i1) + values{2}(i2))/2;
    else
        threshold = 0;
    end
end

% Perform LDA on Two Class Classification
function [alpha,beta,w,threshold,U] = two_feature_lda(num1,num2,features_train,...
                        num_indices_train,num_components)
    i1 = num1+1;
    i2 = num2+1;
    oneft = features_train(:,num_indices_train{i1});
    l1 = length(oneft);
    twoft = features_train(:,num_indices_train{i2});
    l2 = length(twoft);
    features12 = [oneft twoft];
    [U,S12,V12] = svd(features12,'econ');
    U = U(:,1:num_components);
    featuressvd = S12*V12';
    sing1 = featuressvd(1:num_components,1:l1);
    sing2 = featuressvd(1:num_components,l1+1:end);
    mean1 = mean(sing1,2); mean2 = mean(sing2,2);
    Sw = 0;
    for i = 1:l1
        Sw = Sw + (sing1(:,i) - mean1)*(sing1(:,i)-mean1)';
    end
    for i = 1:l2
        Sw = Sw + (sing2(:,i) - mean2)*(sing2(:,i)-mean2)';
    end
    Sb = (mean1-mean2)*(mean1-mean2)';
    [V2,D] = eig(Sb,Sw);
    [~, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    vone = w'*sing1;
    vtwo = w'*sing2;
    if mean(vone) > mean(vtwo)
        w = -w;
        vone = -vone;
        vtwo = -vtwo;
    end
    alpha = sort(vone);
    beta = sort(vtwo);
    
    % Thresholding
    i1 = length(alpha);
    i2 = 1;
    while alpha(i1) > beta(i2)
       i1 = i1-1; i2 = i2 + 1;
    end
    threshold = (alpha(i1) + beta(i2))/2;
end

% LDA Predict
function [labels,labels_pred] = lda_predict(num1,num2,num_indices,features,U,w,threshold)
    % Only works for two variable LDA
    oneftest = features(:,num_indices{num1+1});
    l1 = length(oneftest);
    twoftest = features(:,num_indices{num2+1});
    l2 = length(twoftest);
    labels1 = ones(1,l1);
    labels2 = ones(1,l2)*2;
    labels = [labels1 labels2];
    features_num = [oneftest twoftest];
    features_pca = U'*features_num;
    pred_val = w'*features_pca;
    labels_pred = (pred_val >= threshold) + 1;
    end

% Make Confusion matrix
function con_mat = make_confusion_matrix(labels,labels_pred,check)
    if nargin<3
        check = 0;
    end
    label_list = unique(labels);
    num_categories = length(label_list);
    con_mat = zeros(num_categories);
    for j = 1:length(labels)
        con_mat(labels(j)+check,labels_pred(j)+check) =...
            con_mat(labels(j)+check,labels_pred(j)+check) + 1;
    end
    for n = label_list
        M = sum(con_mat(n+check,:));
        con_mat(n+check,:) = con_mat(n+check,:)./M;
    end
end

% Apply Haar Wavelet Transform
function wavedata = num_wave(imdata)
    [m,n,p] = size(imdata);
    wavedata = zeros(m/2,n/2,p);
    for k = 1:size(imdata,3)
        X = imdata(:,:,k);
        [~,h,v,~] = dwt2(X,'haar');
        wavedata(:,:,k) = rescale(abs(h)) + rescale(abs(v));
    end
end

% Intial Formatting of Data
function [indices, features] = initial_processing(images,labels)
    m = size(images,1); n = size(images,2);
    features = double(reshape(images,m*n,size(images,3)));
    indices = cell(1,length(unique(labels)));
    for i = 1:length(unique(labels))
        indices{i} = find(labels == i-1);
    end
end