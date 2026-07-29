%% Dynamic Features of Yonghe Bridge
%
% Feature Type : Modal Frequencies from Frequency Domain Decomposition (FDD)
% 
% Feature Name
%
% freq_fdd: This dataset contains the modal frequencies of four modes of 9
%           days of measurements. The first eight days are related to the
%           undamaged conditions of the bridge and the last day (Day 12 on
%           July 31, 2008) refers to the damaged condition. The total size
%           of this dataset consists of 216 modal frequencues, where the
%           first 192 samples pertain to the normal condition (192=24x8)
%           and the last 24 samples (193-216) are related to the damaged
%           condition.
%
% If you use this dataset, please cite the following article:

% Daneshvar MH, Sarmadi H (2022) Unsupervised learning-based damage assessment
% of full-scale civil structures under long-term and short-term monitoring.
% Engineering Structures, 256:114059. https://doi.org/10.1016/j.engstruct.2022.114059

% More details of this dataset is also avaliable in the above reference or
% correspond with sarmadi.ipesfp@gmail.com

clc; clear; close all

%% Load Data

% nHe: Number of modal frequencies of the undamaged condition
% nTr: Number of training data
% nTe: Number of test data along with the validation data
% nVd: Number of validation data
% nDa: Number of damaged data (the test data of the damaged condition)
% nTo: Number of all modal frequencies

% Freq_U: Modal frequencies of the undamaged condition
% Freq_D: Modal frequencies of the damaged condition

% X: Training data
% Z: Test data

load('Yonghe_Modal_FDD.mat');

[nMode,nTo] = size(freq_fdd);

nHe = 192;
Freq_U = freq_fdd(:,1:nHe);
Freq_D = freq_fdd(:,nHe+1:nTo);

nTr = floor(0.75*nHe);
X = Freq_U(:,1:nTr);
Z = [Freq_U(:,nTr+1:nHe) Freq_D];
[~,nTe] = size(Z);
nVd = nHe - nTr;
nDa = nTo - nHe;


Title = {'(a)','(b)','(c)','(d)'};
figure
for i = 1:nMode
    subplot(2,2,i)
    f1=semilogy(1:nHe,freq_fdd(i,1:nHe),'LineStyle','none','Marker','.','MarkerSize',10,'Color','b');
    hold on
    line([nHe+0.5 nHe+0.5],[0.25 2],'LineStyle',':','Color','k');
    hold on
    f2=semilogy(nHe+1:nTo,freq_fdd(i,nHe+1:nTo),'LineStyle','none','Marker','x','MarkerSize',6,'Color','r');
    xlabel('Samples'); ylabel('$f$ $(Hz)$','Interpreter','Latex'); title(Title{i});
    if i == 1; set(gca,'XLim',[0 nTo+1],'YLim',[0.25 0.45]); end
    if i == 2; set(gca,'XLim',[0 nTo+1],'YLim',[0.42 0.62]); end
    if i == 3; set(gca,'XLim',[0 nTo+1],'YLim',[0.78 1.02]); end
    if i == 4; set(gca,'XLim',[0 nTo+1],'YLim',[0.97 1.20]); end
    legend([f1,f2],'NC','DC');
end

%% Anomaly Detection by Mahalanobis Distance

dmr = mahal(X',X');
dme = mahal(Z',X');
DIm = [dmr;dme];

figure
f1=plot(1:nTr,DIm(1:nTr),'LineStyle','none','Marker','.','MarkerSize',12,'Color','b');
hold on
line([nTr+0.5 nTr+0.5],[0 50],'LineStyle',':','LineWidth',1.2,'Color','k')
hold on
f2=plot(nTr+1:nHe,DIm(nTr+1:nHe),'LineStyle','none','LineWidth',1.2,'Marker','+','MarkerSize',6,'Color',[0 0.8 0.2]);
hold on
line([nHe+0.5 nHe+0.5],[0 50],'LineStyle',':','LineWidth',1.2,'Color','k')
hold on
f3=plot(nHe+1:nTo,DIm(nHe+1:nTo),'LineStyle','none','LineWidth',1.2,'Marker','x','MarkerSize',7,'Color','r');
hold on
ylabel('DI'); xlabel('Samples')
legend([f1,f2,f3],'UDC - Training','UDC - Validation','DC','location','northwest');
set(gca,'XLim',[0 nTo+1])
