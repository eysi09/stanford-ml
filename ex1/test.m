v = ones(7,1)
w = [1;2;3;4;5;6;7];
z = 0;
for i = 1:7
  z = z + v(i) * w(i);
end
z