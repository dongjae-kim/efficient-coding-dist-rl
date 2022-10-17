function [scaleBetas, scaleBetas_lb, scaleBetas_ub]= EstimateScaleFactors_sigmoid(perCell, utilityAxisPerCell)
%

nCells = size(perCell,1);
nTrials = size(perCell,3);

scaleBetas = nan(nCells,3);
scaleBetas_lb = nan(nCells,3);
scaleBetas_ub = nan(nCells,3);

% scaleFactNeg = nan(nCells,1);
% scaleFactNegSE = nan(nCells,1);
% scaleFactNegN = nan(nCells,1);
% scaleFactPos = nan(nCells,1);
% scaleFactPosSE = nan(nCells,1);
% scaleFactPosN = nan(nCells,1);
for iCell=1:nCells
    negpart = utilityAxisPerCell(iCell,:) <= 0;
    pospart = utilityAxisPerCell(iCell,:) >= 0;
    allpart = logical(ones(1,7));
    X_all = repmat(-utilityAxisPerCell(iCell,allpart), [1 1 nTrials]);
    Y_all = -perCell(iCell,allpart,:);
    
    
    % Fitting sigmoid function
    modelfun = @(beta,xvals) sig_function(beta, xvals);
    opts = statset('nlinfit');
    opts.Display = 'off';
    opts.MaxIter = 1e4;
%     [beta,R,J] = nlinfit(lin(X_all),lin(Y_all),modelfun,[rand;rand;rand],opts); 
    [beta,resnorm,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(modelfun,[rand;rand;rand],lin(X_all(find(~isnan(Y_all)))),lin(Y_all(find(~isnan(Y_all)))),[0,0,-100]',[inf,1,100]');
    ci = nlparci(beta,residual,'jacobian',jacobian,'alpha',0.975);
    %     scaleFactNegSE(iCell) = NaN;
    scaleBetas(iCell,:) = beta'; %1 * 3
    scaleBetas_lb(iCell,:) = ci(:,1)';
    scaleBetas_ub(iCell,:) = ci(:,2)';
            
end

end

