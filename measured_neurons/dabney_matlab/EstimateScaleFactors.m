function [scaleFactNeg, scaleFactNegSE, scaleFactNegN, scaleFactPos, scaleFactPosSE, scaleFactPosN] = EstimateScaleFactors(perCell, utilityAxisPerCell, functionType)
%

nCells = size(perCell,1);
nTrials = size(perCell,3);

scaleFactNeg = nan(nCells,1);
scaleFactNegSE = nan(nCells,1);
scaleFactNegN = nan(nCells,1);
scaleFactPos = nan(nCells,1);
scaleFactPosSE = nan(nCells,1);
scaleFactPosN = nan(nCells,1);
for iCell=1:nCells
    negpart = utilityAxisPerCell(iCell,:) <= 0;
    pospart = utilityAxisPerCell(iCell,:) >= 0;
    if any(negpart)
        X = repmat(-utilityAxisPerCell(iCell,negpart), [1 1 nTrials]);
        Y = -perCell(iCell,negpart,:);
        
        if strcmp(functionType, 'hill')
            modelfun = @(beta, xvals) hillfunc(xvals, beta, 10);
            [beta,R,J] = nlinfit(lin(X),lin(Y),modelfun,rand);
            ci = nlparci(beta,R,'jacobian',J);
            scaleFactNegSE(iCell) = ci(2)-beta;
            scaleFactNeg(iCell) = beta;
        elseif strcmp(functionType, 'linear')
            [beta,~,stats] = glmfit(lin(X), lin(Y), 'normal', 'constant', 'off');
            scaleFactNegSE(iCell) = stats.se;
            scaleFactNeg(iCell) = beta;
        elseif strcmp(functionType, 'sigmoid')
            modelfun = @(beta,xvals) sig_function(beta, xvals)
            
            [beta,R,J] = nlinfit(lin(X),lin(Y),modelfun,[rand;rand]);
        end

        scaleFactNegN(iCell) = sum(~isnan(lin(perCell(iCell,negpart,:))));
    end
    if any(pospart)
        X = repmat(utilityAxisPerCell(iCell,pospart), [1 1 nTrials]);
        Y = perCell(iCell,pospart,:);
        
        if strcmp(functionType, 'hill')
            modelfun = @(beta, xvals) hillfunc(xvals, beta, 10);
            [beta,R,J] = nlinfit(lin(X),lin(Y),modelfun,rand);
            ci = nlparci(beta,R,'jacobian',J);
            scaleFactPosSE(iCell) = ci(2)-beta;
            scaleFactPos(iCell) = beta;
        elseif strcmp(functionType, 'linear')
            [beta,~,stats] = glmfit(lin(X), lin(Y), 'normal', 'constant', 'off');
            scaleFactPosSE(iCell) = stats.se;
            scaleFactPos(iCell) = beta;
        end

        scaleFactPosN(iCell) = sum(~isnan(lin(perCell(iCell,pospart,:))));
    end
end

end

