function returning_=sig_function(beta,xvals)

returning_=[];
for i =1:1:length(xvals)
    returning_ = [returning_ ;  (beta(1)/(1+exp(-beta(2)*xvals(i)))) + beta(3)];
end
