function [scaleBetas, scaleBeta, scaleBetas_lb, scaleBetas_ub, scaleBeta_lb, scaleBeta_ub] = EstimateScaleFactors7_sigmoid(perCell, utilityAxisPerCell, halfOfDataForAsymmetry)
% estimate scale factors in 7 disjoint subsets of the data, in order to get 
% a measure of variance

nCellsUsed = size(perCell,1);
nRwdMags = size(perCell,2);


nFolds = 7;
nPerFold = floor(sum(~isnan(perCell),3) / nFolds);
usedUp = false(size(perCell));
usedUp(isnan(perCell)) = true;
scaleBetas = nan(nFolds, nCellsUsed, 3);
scaleBetas_lb = nan(nFolds, nCellsUsed, 3);
scaleBetas_ub = nan(nFolds, nCellsUsed, 3);
for iFold = 1:nFolds
    perCellF = nan(nCellsUsed, nRwdMags, ceil(size(perCell,3)/nFolds));
    for iCell=1:nCellsUsed
        for iRwdMag = 1:nRwdMags
            % use a random subset of the indices of ~usedUp(iCell,iRwdMag,:)
            toUse = randsample(find(~usedUp(iCell,iRwdMag,:)), nPerFold(iCell,iRwdMag));
            usedUp(iCell,iRwdMag,toUse) = true;
            perCellF(iCell,iRwdMag,1:length(toUse)) = perCell(iCell,iRwdMag,toUse);
        end
    end
    [scaleBetas(iFold,:,:), scaleBetas_lb(iFold,:,:), scaleBetas_ub(iFold,:,:)]= EstimateScaleFactors_sigmoid(perCellF, utilityAxisPerCell);
%     [scaleFactsNeg(iFold,:), ~, ~, scaleFactsPos(iFold,:), ~, ~] = ...
%         EstimateScaleFactors(perCellF, utilityAxisPerCell, functionType);
end
[scaleBeta, scaleBeta_lb, scaleBeta_ub]= EstimateScaleFactors_sigmoid(perCell .* halfOfDataForAsymmetry, utilityAxisPerCell);


% scaleFactNeg = mean(scaleFactsNeg,1)';
% scaleFactPos = mean(scaleFactsPos,1)';
% sfNegSEM = sem(scaleFactsNeg,1)';
% sfPosSEM = sem(scaleFactsPos,1)';
% 
% asym = scaleFactPos ./ (scaleFactPos + scaleFactNeg);
% 
% asyms = scaleFactsPos ./ (scaleFactsPos + scaleFactsNeg);
% asymSE = sem(asyms,1);
% [~,asym_pvals] = ttest(asyms-nanmean(nanmean(asyms,1),2));


end

