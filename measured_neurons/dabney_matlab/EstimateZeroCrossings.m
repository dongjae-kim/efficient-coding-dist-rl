function [zeroCrossings] = EstimateZeroCrossings(perCell, utilityAxis)
%

nCells = size(perCell,1);

zeroCrossings = nan(nCells,1);
for iCell=1:nCells
    xData = repmat(1:7, [1 1 size(perCell,3)]);
    yData = perCell(iCell,:,:);
    validpart = ~isnan(yData);
    
    k = 0.5:1:7.5;
    critvals = nan(length(k),1);
    for iK = 1:length(k)
        ix_above = xData(validpart) > k(iK);
        ix_below = xData(validpart) < k(iK);
        y_valid = yData(validpart);

        %% proportion method
%         nPosAboveZero = sum(y_valid(ix_above)>0);
%         nPosBelowZero = sum(y_valid(ix_below)>0);
%         nNegAboveZero = sum(y_valid(ix_above)<0);
%         nNegBelowZero = sum(y_valid(ix_below)<0);
%         critvals(iK) = (nPosAboveZero + nNegBelowZero) ./ (nPosAboveZero + nPosBelowZero + nNegAboveZero + nNegBelowZero); % take the *proportion* of points on the correct side

        %% total count method
        critvals(iK) = sum(y_valid(ix_above)>0) + sum(y_valid(ix_below)<0);  % *count* the number of points above and below
    end
    
    critvals = critvals + 0.001*randn(size(critvals));  % avoid ties
    [~,mcv] = max(critvals);
    
    zc = k(mcv); % we define the zero-crossing as the point that makes the sum of the points above it as big as possible, and the sum of the points below it as small as possible
    
    if zc < 1
        zeroCrossings(iCell) = utilityAxis(1) + 0.01; % we arbitrarily cap at the minimum reward magnitude
    elseif zc > 7
        zeroCrossings(iCell) = utilityAxis(end) - 0.1; % totally arbitrary, because there's no way to know how to extrapolate off the edge
    else
        w = abs(diff([critvals(mcv-1) critvals(mcv) critvals(mcv+1)]));
        w = (1./w) / sum(1./w);

        zeroCrossings(iCell) = w(1)*utilityAxis(zc-0.5) + w(2)*utilityAxis(zc+0.5);  % convert the zero-crossing into utility space    
        % note that we weight it by how close the critical value is to its neighbors on either side
    end
    
end

end

