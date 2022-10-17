function returning_ = logarithm(beta, xvals)

returning_=[];
for i =1:1:length(xvals)
    returning_ = [returning_ ;  log(xvals(i)-beta(2))/log(beta(1))];
end

