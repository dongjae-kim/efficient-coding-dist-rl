function [indices] = timesToIndices(times, startEnd, resolution)
% given a time window [startEnd(1) startEnd(2)], sampled at "resolution"
% converts the range [times(1) times(2)] to a list of indices in that time window

assert(times(1) >= startEnd(1))
assert(times(2) <= startEnd(2))

indices = (1 + (times(1) - startEnd(1)) / resolution) : (times(2) - startEnd(1)) / resolution;

end

