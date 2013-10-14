function W = orthonorm(M, N, k, scale, gpuState)

W = gpuState.randn(M, N, k);

for i = 1:k
   % Subtract off previous vectors
   for j = 1:i-1
      projCoeff = sum(W(:,:,i).*W(:,:,j), 2);
      W(:,:,i) = W(:,:,i) - bsxfun(@times, projCoeff, W(:,:,j));
   end
   % Normalize rows to have unit L2 norm
   W(:,:,i) = bsxfun(@rdivide, W(:,:,i), sqrt(sum(W(:,:,i).*W(:,:,i), 2)));
end

W = scale*W;

end

