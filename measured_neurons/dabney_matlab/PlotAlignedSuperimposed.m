function [] = PlotAlignedSuperimposed(meanOverTrials, utilityAxisPerCell, asymColor)
%

nCells = size(meanOverTrials,1);


for iCell=1:nCells
    posPart = utilityAxisPerCell(iCell,:) > 0;
    plot([utilityAxisPerCell(iCell,~posPart) 0], [meanOverTrials(iCell,~posPart) 0], '-', ...
        'Color', asymColor(iCell,:), 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'none', 'LineWidth', 3)
    hold on
    plot([0 utilityAxisPerCell(iCell,posPart)], [0 meanOverTrials(iCell,posPart)], '-', ...
        'Color', asymColor(iCell,:), 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'none', 'LineWidth', 3)
    
%     text(utilityAxisPerCell(iCell,end) + 0.05, meanOverTrials(iCell,end), num2str(iCell))
end
xlabel('reward magnitude minus zero-crossing')
ylabel('firing rate above baseline (Hz)')


end

