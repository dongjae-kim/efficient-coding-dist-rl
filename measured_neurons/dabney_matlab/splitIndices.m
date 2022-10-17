function [halfOfDataForReversalPoints, halfOfDataForAsymmetry] = splitIndices(perCell)
%

nCellsUsed = size(perCell,1);
nRwdMags = size(perCell,2);

halfOfDataForReversalPoints = nan(size(perCell));
for iCell=1:nCellsUsed
    for iRwdMag = 1:nRwdMags
        linInds = find(~isnan(squeeze(perCell(iCell, iRwdMag, :))));
        toUse = randsample(linInds, floor(length(linInds)/2));
        halfOfDataForReversalPoints(iCell, iRwdMag, toUse) = 1;
    end
end
halfOfDataForAsymmetry = nan(size(perCell));
halfOfDataForAsymmetry(isnan(halfOfDataForReversalPoints)) = 1;

end

