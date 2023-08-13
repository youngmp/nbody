function dy = getrc(t,Y)
global perorb myfn mytime dt kappa


myt = t;
XX = floor(myt/dt);
xrem = myt-XX*dt;

if XX==10000-1
myY = perorb(XX+1,:);
else
myY = perorb(XX+1,:) + (perorb(XX+2,:)-perorb(XX+1,:))*(xrem/dt);
end


dy = (numericaljacobian(myfn,myY)-kappa*eye(length(Y)))*Y;
dy = dy(:);