function dy = monodromyfn(t,Y)
global perorb myfn mytime dt

Y = reshape(Y,sqrt(length(Y)),sqrt(length(Y)));

myt = t;
XX = floor(myt/dt);
xrem = myt-XX*dt;


if XX==10000-1
myY = perorb(XX+1,:);
else
myY = perorb(XX+1,:) + (perorb(XX+2,:)-perorb(XX+1,:))*(xrem/dt);
end


dy = numericaljacobian(myfn,myY)*Y;
dy = dy(:);


t;