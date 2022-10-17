
clear all, close all
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
[~, PSTH, nCells, animalNames, ~, ~, baselines] = LoadVarMagData(psthWindow, psthResolution, ...
    smoothingTimeConst, baselineWindow);
maxTrials = size(PSTH,3);

%% which animals to use cells from
whichCellsToUse = true(nCells,1);
nCellsUsed = sum(whichCellsToUse);
animalNamesUsed = animalNames(whichCellsToUse);
baselines = baselines(whichCellsToUse,:,:);

%% make some useful variables
dataUnexpected = mean(PSTH(whichCellsToUse, 1:7, :, timesToIndices(responseWindowUnexp, ...
    psthWindow, psthResolution)), 4);
dataExpected = mean(PSTH(whichCellsToUse, 8:14, :, timesToIndices(responseWindowExp, ...
    psthWindow, psthResolution)), 4);
perCell = cat(3, dataExpected,dataUnexpected - mean(mean(nanmean(dataUnexpected,3),1),2)...
    + mean(mean(nanmean(dataExpected,3),1),2));

mpc = nanmean(perCell,3); 
utilityAxis = mean(mpc,1); % the empirical utility function

%% divide the data into two halves: one to compute the zero-crossings, and one to compute the slopes
[halfOfDataForReversalPoints, halfOfDataForAsymmetry] = splitIndices(perCell);


%% 1) estimate a zero-crossing for each cell, in a model-free way, and align each cell 
%% to its zero-crossing
zeroCrossings_rev = EstimateZeroCrossings(perCell .* halfOfDataForReversalPoints, utilityAxis);

zeroCrossings_asym = EstimateZeroCrossings(perCell .* halfOfDataForAsymmetry, utilityAxis);
utilityAxisPerCell_asym = utilityAxis - zeroCrossings_asym;

zeroCrossings_all = EstimateZeroCrossings(perCell, utilityAxis);
utilityAxisPerCell_all = utilityAxis - zeroCrossings_all;


%% 2) estimate scale factors separately for positive and negative domains, using parametric
%% fit to hill function or linear function.
%% try a trick where we partition the trials into 7 random groups, to get SEMs
[scaleFactNeg_asym, scaleFactPos_asym, asym, asymSE, asym_pvals, sfNegSEM, sfPosSEM, asyms] = ...
    EstimateScaleFactors7(perCell, utilityAxisPerCell_asym, functionType, halfOfDataForAsymmetry);
asymM_asym = scaleFactPos_asym ./ (scaleFactNeg_asym + scaleFactPos_asym);

[scaleFactNeg_all, scaleFactPos_all, ~, asymSE_all, ~, ~, ~, ~] = ...
    EstimateScaleFactors7(perCell, utilityAxisPerCell_all, functionType, ones(size(perCell)));
asymM_all = scaleFactPos_all ./ (scaleFactNeg_all + scaleFactPos_all);


%% make some color scales for the two variables (asymmetry and zero-crossing)
asymCmap = [linspace(0,1,101)' zeros(101,1) linspace(1,0,101)'];
asymColor = asymCmap(1+max(0,min(100,round(100*asymM_asym))),:);
asymColor(isnan(asymM_asym),:) = 0;

zeroCrossings_all(isnan(asymM_asym)) = nan; % don't normalize in any cells we're not going to plot
zcSorted = sort(zeroCrossings_all);
zcNorm = (zeroCrossings_all - min(zeroCrossings_all)) ./ (zcSorted(end-1)-min(zeroCrossings_all));
zcColor = asymCmap(1+max(0,min(100,round(100*zcNorm))),:);


%% 3) plot all cells re-aligned and re-scaled (arbitrarily scaled to only negative domain; 
%% could equivalently scale them only to positive domain)
nin = ~isnan(asymM_asym);  % only plot the ones that aren't NaNs
figure, PlotAlignedSuperimposed(nanmean(perCell(nin,:,:) ./ scaleFactNeg_all(nin),3), utilityAxisPerCell_all(nin,:), zcColor(nin,:))


%% 4) plot each cell individually (uses magic number indices)
if nCellsUsed == nCells
    figure,
    cellsToPlot = [25 33 1 17 15 29 26 19 21];
    
    minY = min(min(nanmean(perCell,3)));
    maxY = max(max(nanmean(perCell,3)));
    for iCellI=1:length(cellsToPlot)
        subplot(3,3,iCellI)
        iCell = cellsToPlot(iCellI);
        negPart = utilityAxisPerCell_all(iCell,:) <= 0;
        posPart = utilityAxisPerCell_all(iCell,:) > 0;
        negX = utilityAxisPerCell_all(iCell,negPart);
        posX = utilityAxisPerCell_all(iCell,posPart);
        
        minUAx = -8;
        maxUAx = 13;
        if strcmp(functionType, 'hill')
            plot(minUAx:0.01:0, -hillfunc(-(minUAx:0.01:0), scaleFactNeg_all(iCell), 200), 'b-')
            plot(0:0.01:maxUAx, hillfunc(0:0.01:maxUAx, scaleFactPos_all(iCell), 200), 'r-')
        elseif strcmp(functionType, 'linear')
            plot(minUAx:0.01:0, scaleFactNeg_all(iCell) * (minUAx:0.01:0), 'b-'), hold on
            plot(0:0.01:maxUAx, scaleFactPos_all(iCell) * (0:0.01:maxUAx), 'r-')
        end
        errorbar(negX, nanmean(perCell(iCell,negPart,:),3), ...
            sem(perCell(iCell,negPart,:),3), 'b.')
        errorbar(posX, nanmean(perCell(iCell,posPart,:),3), ...
            sem(perCell(iCell,posPart,:),3), 'r.')
        
        plot([minUAx maxUAx], [0 0], 'k:')
        plot([0 0], [minY maxY], 'k:')
        axis([minUAx,maxUAx,minY,maxY])
        title(['cell ' num2str(iCell)])
        
        rectangle('Position',[-8 11 2 2], 'FaceColor', asymColor(iCell,:))
    end
end

%% 5) plot the *ratio* of +ve and -ve slopes for each cell
[~,sortOrder] = sort(asymM_all);

cellx = 1:nCellsUsed;
figure, 
for iCell=1:nCellsUsed
    if ~isnan(asymSE_all(sortOrder(iCell)))
        errorbar(iCell, ...
            asymM_all(sortOrder(iCell)), ...
            asymSE_all(sortOrder(iCell)), '.', ...
            'Color', asymColor(sortOrder(iCell),:))
        hold on
    end
end
meanDiff = nanmean(asymM_all);
plot([0 1+nCellsUsed], [meanDiff meanDiff], 'k--')
plot(cellx(asym_pvals(sortOrder)<0.05), ones(1,sum(asym_pvals(sortOrder)<0.05)), 'k*')
legend({'each cell', 'mean across cells'})
ylabel('\eta^+ / (\eta^+ + \eta^-)')
xlabel('cell index')
xlim([0 1+nCellsUsed])

S = load('distSims');
[~,ix] = sort(S.distSims(1,:));
S.distSims = S.distSims(:,ix);

%% 6) plot zero-crossings against kinks
couldEstimate = scaleFactNeg_asym~=0 & scaleFactPos_asym~=0 & ~isnan(scaleFactPos_asym) & ~isnan(scaleFactNeg_asym);  % don't use the ones we didn't have any data points for
figure, 
for i=1:10
    plot(S.distSims(1,:)', S.distSims(1+i,:)', '-', 'Color', [0.7 0.7 0.7])
    hold on
end
for iCell=1:nCellsUsed
    if couldEstimate(iCell)
        scatter(asymM_asym(iCell), zeroCrossings_rev(iCell), 30, 'MarkerFaceColor', asymColor(iCell,:), 'MarkerEdgeColor', 'none')
    end
end
[~,sortix] = sort(utilityAxis);
set(gca, 'YTick', utilityAxis(sortix), 'YTickLabel', conditionNames(sortix))
xlabel('\eta^+ / (\eta^+ + \eta^-)')
ylabel('reversal point')
[beta, ~, stats] = glmfit(asymM_asym, zeroCrossings_rev);
R = corr(asymM_asym, zeroCrossings_rev, 'rows', 'complete');
minAS = min(asymM_asym);
maxAS = max(asymM_asym);
hold on, plot([0 1], beta(1) + beta(2)*[0 1], 'k-')
title(['R = ' num2str(R) ', p = ' num2str(stats.p(2))])
axis([0 1 -4 8])
p = mean(sum(~isnan(perCell),3),1); p = p / sum(p);
utMean = ((sum(juiceAmounts .* p) - juiceAmounts(5)) / (juiceAmounts(6) - juiceAmounts(5))) ...
    * (utilityAxis(6) - utilityAxis(5)) + utilityAxis(5); % the average reward magnitude, in utility space
plot([0 1], utMean * [1 1], 'k--')

%%
%%
%% 
%% above part is the code used by Dabney et al., 2020
%% save fit results

save('dabney_fit.mat','asymM_all','perCell',...
    'scaleFactNeg_all','scaleFactPos_all','zeroCrossings_all','utilityAxis')



%% data axis and utility axis

ys = [];
xs = [];
for j = 1: 1: 7
    ys = [ys;ones(40,1)*juiceAmounts(j)];
    xs = [xs;mpc(:,j)];
end
mdl_lm = fitlm(xs,ys);
betas = mdl_lm.Coefficients.Estimate;

save('dabney_utility_fit.mat','betas','xs','ys')

