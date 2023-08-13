function dy = monodromyfn(t,Y)
global perorb myfn mytime dt kappa g1 g2 g3 g4 g5 g6 g7 g8 numberderivs


myt = t;
XX = floor(myt/dt);
xrem = myt-XX*dt;

if XX==10000-1
myY = perorb(XX+1,:);
g1int = g1(:,XX+1);
g2int = g2(:,XX+1);
g3int = g3(:,XX+1);
g4int = g4(:,XX+1);
g5int = g5(:,XX+1);
g6int = g6(:,XX+1);
g7int = g7(:,XX+1);
g8int = g8(:,XX+1);
else
myY = perorb(XX+1,:) + (perorb(XX+2,:)-perorb(XX+1,:))*(xrem/dt);
g1int = g1(:,XX+1) + (g1(:,XX+2)-g1(:,XX+1))*(xrem/dt);
g2int = g2(:,XX+1) + (g2(:,XX+2)-g2(:,XX+1))*(xrem/dt);
g3int = g3(:,XX+1) + (g3(:,XX+2)-g3(:,XX+1))*(xrem/dt);
g4int = g4(:,XX+1) + (g4(:,XX+2)-g4(:,XX+1))*(xrem/dt);
g5int = g5(:,XX+1) + (g5(:,XX+2)-g5(:,XX+1))*(xrem/dt);
g6int = g6(:,XX+1) + (g6(:,XX+2)-g6(:,XX+1))*(xrem/dt);
g7int = g7(:,XX+1) + (g7(:,XX+2)-g7(:,XX+1))*(xrem/dt);
g8int = g8(:,XX+1) + (g8(:,XX+2)-g8(:,XX+1))*(xrem/dt);
end

[f2] = symsderivs_higherorder(myY(:),g1int,g2int,g3int,g4int,g5int,g6int,g7int,g8int,numberderivs);


dy = (numericaljacobian(myfn,myY)-numberderivs*kappa*eye(length(Y)))*Y + f2;
dy = dy(:);
