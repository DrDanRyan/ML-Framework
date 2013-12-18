function conv_speed_test(layer, x, N)
for i = 1:N
   y = layer.feed_forward(x, true);
   %[g, s] = layer.backprop(x, y, dLdy);
end
end

