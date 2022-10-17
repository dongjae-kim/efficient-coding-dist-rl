function [spikeRaster, PSTH, nCells, animalNames, nTrials, ITIs, baselines, pstc] = LoadVarMagData_spikes(psthWindow, psthResolution, smoothingTimeConst, baselineWindow)
%

%% location of data
dataPath = 'variable-reward/Light-ID dopamine';
fileList = dir([dataPath '/*.mat']); fileList = {fileList(:).name};
nCells = length(fileList);

%% trial parameters
interestingTrialTypes = [1 2 3 4 5 6 7   8 9 10 11 12 13 14];  % 1-7 are unexpected; 8-14 are expected
nTrialTypes = length(interestingTrialTypes);
maxTrialsPerType = 60;  % most trials that any cell has

%% PSTH parameters
psthLength = 1 + ceil((psthWindow(2) - psthWindow(1)) / psthResolution); % number of bins in the PSTH

%% smoothing kernel
smoothingFunc = @(t) (1 - exp(-t)).*exp(-t/smoothingTimeConst);
smoothingKernel = smoothingFunc(0:psthResolution:(2.5*smoothingTimeConst));
smoothingKernel = smoothingKernel / sum(smoothingKernel);
smLength = length(smoothingKernel);


%% pre allocating variables
spikeRaster = nan(nCells, nTrialTypes, maxTrialsPerType, psthLength);  % peri-stimulus time counts, including all cells
PSTC = nan(nCells, nTrialTypes, maxTrialsPerType, psthLength);  % peri-stimulus time counts, including all cells
nTrials = nan(nCells, 1);  % how many actual trials per cell
animalNames = cell(nCells,1);
sessionDates = cell(nCells,1);
ITIs = nan(nCells,nTrialTypes, maxTrialsPerType);
dat = [];
iTrialOfTypes = zeros(nTrialTypes,1);
for iCell=1:nCells
    S = load([dataPath '/' fileList{iCell}]);
    dat = [dat;S.events.trialType];
    iTrialOfType = ones(nTrialTypes,1);   % keep track (separately per cell) of how many trials of each type have happened
    nTrials(iCell) = length(S.events.odorOn);
    fnToks = strsplit(fileList{iCell}, '_');
    animalNames{iCell} = fnToks{1};
    sessionDates{iCell} = fnToks{2};
    
    for iTrial=1:nTrials(iCell)
        odorTime = S.events.odorOn(iTrial);  % an odor-on event
        
        %% take a temporal window around the odor onset time
        windowIndices = S.responses.spike > odorTime + psthWindow(1) & S.responses.spike < odorTime + psthWindow(2);
        
        %% find the spikes that happened in that window, and map their times to PSTH indices
        spikeTimes = S.responses.spike(windowIndices);
        spikePsthIndices = 1 + round((spikeTimes - odorTime) / psthResolution) - psthWindow(1) / psthResolution;
        
        %% find the trial type
        [~,trialTypeIx] = ismember(S.events.trialType(iTrial), interestingTrialTypes);
        
        %% increment the PSTH with these spikes
        if trialTypeIx >= 1 && trialTypeIx <= nTrialTypes
            if iTrial > 1  % record the ITI for this trial (i.e., the time since the start of the previous trial)
                ITIs(iCell, trialTypeIx, iTrialOfType(trialTypeIx)) = odorTime - S.events.odorOn(iTrial-1); 
            end
            
            PSTC(iCell, trialTypeIx, iTrialOfType(trialTypeIx), :) = zeros(1,1,1,psthLength);
            spikeRaster(iCell, trialTypeIx, iTrialOfType(trialTypeIx), :) = zeros(1,1,1,psthLength);

            for iSpike = 1:length(spikePsthIndices)
                tbs = spikePsthIndices(iSpike);  % time bin start
                tbe = min(tbs+smLength-1, psthLength);
                PSTC(iCell, trialTypeIx, iTrialOfType(trialTypeIx), tbs:tbe) = ...
                    PSTC(iCell, trialTypeIx, iTrialOfType(trialTypeIx), tbs:tbe) + ...
                    shiftdim(smoothingKernel(1:(tbe-tbs+1)), -2);  % smooth each spike as we count it
                
                spikeRaster(iCell, trialTypeIx, iTrialOfType(trialTypeIx), spikePsthIndices(iSpike)) = ...
                    spikeRaster(iCell, trialTypeIx, iTrialOfType(trialTypeIx), spikePsthIndices(iSpike)) + 1;  % also put the spike into the raster
            end

            iTrialOfType(trialTypeIx) = iTrialOfType(trialTypeIx) + 1;
        end
    end
    
    iTrialOfTypes = iTrialOfTypes + iTrialOfType;
end

PSTH = (1000 / psthResolution) * PSTC; % convert spikes/(psthResolution ms) to spikes/s
baselines = nanmean(PSTH(:,:,:,timesToIndices(baselineWindow, psthWindow, psthResolution)), 4);
pstc = PSTH;
pstc = convn(pstc, shiftdim(smoothingKernel, -2), 'same');  % apply smoothing
PSTH = PSTH - baselines;
PSTH = convn(PSTH, shiftdim(smoothingKernel, -2), 'same');  % apply smoothing


end

