
% clear all, close all
rng(1)

%% windows
psthWindow = [-1000 3000];  % in units of milliseconds
psthResolution = 1;  % in units of milliseconds
psthLength = 1 + ceil((psthWindow(2) - psthWindow(1)) / psthResolution); % number of bins in PSTH
psthTimes = linspace(psthWindow(1), psthWindow(2), psthLength);  % relative time of each bin in PSTH
baselineWindow = [-1000 0];  % in units of milliseconds
responseWindowUnexp = [200 600];
responseWindowExp = [1700 2100];

%% smoothing kernel
smoothingTimeConst = 20; 
smoothingFunc = @(t) (1 - exp(-t)).*exp(-t/smoothingTimeConst); 
smoothingKernel = smoothingFunc(0:psthResolution:(2.5*smoothingTimeConst));
smoothingKernel = smoothingKernel / sum(smoothingKernel);

%% trial parameters
conditionNames = {'0.1uL','0.3uL','1.2uL','2.5uL','5uL','10uL','20uL'};
juiceAmounts = [0.1 0.3 1.2 2.5 5 10 20];
nRwdMags = length(juiceAmounts);

%% analysis options parameters
functionType = 'linear';  % can be 'linear' or 'hill'

%% load data
[~, PSTH, nCells, animalNames, ~, ~, baselines, PSTC] = LoadVarMagData_spikes(psthWindow, psthResolution, ...
    smoothingTimeConst, baselineWindow);
maxTrials = size(PSTH,3);

%% which animals to use cells from
whichCellsToUse = true(nCells,1);
nCellsUsed = sum(whichCellsToUse);
animalNamesUsed = animalNames(whichCellsToUse);
baselines = baselines(whichCellsToUse,:,:);

%% make some useful variables

% baselineCorrection = 1;
% 
% if baselineCorrection
%     spikes = PSTH;
% else
spikes = PSTC;
% end

dataUnexpected_All = mean(spikes(whichCellsToUse, 1:1:7, :, timesToIndices(responseWindowUnexp, ...
    psthWindow, psthResolution)), 4);
dataExpected_All = mean(spikes(whichCellsToUse, 8:1:14, :, timesToIndices(responseWindowExp, ...
    psthWindow, psthResolution)), 4);


perCell = cat(3, dataExpected_All,dataUnexpected_All - mean(mean(nanmean(dataUnexpected_All,3),1),2)...
    + mean(mean(nanmean(dataExpected_All,3),1),2));

dat_cell = cell(1,40);
for i = 1:1:40
    datss = [];
    for j = 1 :1 : 7
        datss = [datss squeeze(perCell(i,j,:))];
    end
    dat_cell{1,i} = datss;
end

dat = struct('dat', dat_cell);
save('dat_eachneuron.mat', 'dat')


% if baselineCorrection
spikes = PSTH;

dataUnexpected_All = mean(spikes(whichCellsToUse, 1:1:7, :, timesToIndices(responseWindowUnexp, ...
    psthWindow, psthResolution)), 4);
dataExpected_All = mean(spikes(whichCellsToUse, 8:1:14, :, timesToIndices(responseWindowExp, ...
    psthWindow, psthResolution)), 4);


perCell = cat(3, dataExpected_All,dataUnexpected_All - mean(mean(nanmean(dataUnexpected_All,3),1),2)...
    + mean(mean(nanmean(dataExpected_All,3),1),2));

dat_cell = cell(1,40);
for i = 1:1:40
    datss = [];
    for j = 1 :1 : 7
        datss = [datss squeeze(perCell(i,j,:))];
    end
    dat_cell{1,i} = datss;
end

dat = struct('dat', dat_cell);
save('dat_eachneuron_bc.mat', 'dat')

