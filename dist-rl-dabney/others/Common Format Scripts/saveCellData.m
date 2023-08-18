% SAVECELLDATA: saveCellData(filename, events, responses, header)
%
% This is a helper function to save the event structure and spike time
% stamps to a single .mat file for future analysis.
%
% Inputs: (required)
% - filename: location of a target file. Entering 0 will bring up a ui box.
% - events: event structure for the session
% - response: structure for the different responses. Each type (spike,
% lick, lever press) gets a field with  vector of absolute timestamps.
% Inputs: (optional)
% - header: text used to identify the data.  This could include the
%           experiment name, animal, date, cell, and location of original 
%           data.
% - batchfilename: location of a file to add the cell name for future
%           batch analysis of data

% Vinod Rao

function saveCellData(filename,events,responses,header,batchfilename)

%% Check inputs
if ~isequal(filename, 0) 
    ind = strfind(filename,filesep);
    dirname = filename(1:ind(end)); %this is the text up to the slash
    if ~exist(dirname,'dir')
        error('Directory name does not exist.  Please create the directory.');
    end
end
if ~isstruct(events)
    error('events must be a structure.');
end
if ~isstruct(responses) 
    error('responses must be a structure.')
end
if nargin > 2
    if ~isempty(header)
        if ~ischar(header)
            error('header must be a string.');
        end
    else
        header = filename;
    end
end
if nargin > 3
    saveBatch = 1;
    if ~isequal(batchfilename, 0) 
        ind = strfind(batchfilename,filesep);
        batchpathname = batchfilename(1:ind(end)); %this is the text up to the slash
        batchfilename = batchfilename(ind(end)+1:end);
        if ~exist(batchpathname,'dir')
            error('Directory name does not exist.  Please create the directory.');
        end
    else
        [batchfilename, batchpathname] = uigetfile('*.txt','Select Batch File');
        if batchfilename==0
           error('Files not saved.  Please select a batchfile.')
        end
    end
end

%% now save the file
if filename ~= 0
    if exist('header','var')
        save(filename, 'events','responses', 'header');
    else
        save(filename, 'events','responses');
    end
else
    if exist('header','var')
        [filename, pathname] = uiputfile({'events','responses','header'});
    else
        [filename, pathname] = uiputfile({'events','responses'});
    end
    filename = [pathname filesep filename];
end

%% now save to a batch file
% this writes the filename (including path) to a batchfile
if saveBatch
    fid = fopen([batchpathname filesep batchfilename],'a');
    fprintf(fid,'%s\n',filename);
    fclose(fid);
end

end