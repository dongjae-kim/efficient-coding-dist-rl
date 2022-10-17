function [l] = lin(x)
% linearize x. useful if you've just done an indexing operation and want to linearize inline.
% because for some reason this syntax is not valid in matlab: x(:,1:5,:)(:)


l = x(:);


end