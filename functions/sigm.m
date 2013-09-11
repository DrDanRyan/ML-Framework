function value = sigm(z)
   if z >= 0
      value = 1./(1 + exp(-z));
   else 
      value = exp(z)./(1 + exp(z));
   end
end