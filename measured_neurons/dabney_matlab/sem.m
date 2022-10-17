function [ y ] = sem( x, d )
% input data x.
% calculates standard error of the mean along the dth dimension.
% returns an array with one less dimension than the input.
% ignores nans.

if nargin < 2
    if isvector(x)
        d = find(size(x) > 1);
    else
        error('must specify dimension for multidimensional input')
    end
end


y = squeeze(nanstd(x,0,d) ./ sqrt(sum(~isnan(x), d)));  % only count the non-nan values for sqrt(N)

end

