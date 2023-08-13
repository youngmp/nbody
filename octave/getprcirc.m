function dy = getprcirc(t,Y)
global perorb myfn mytime dt kappa myorb PRCORIRC myorder zfn  numberderivs g1 g2 g3 g4 g5 g6 g7 g8 z0 z1 z2 z3 z4 z5 z6 z7 z8 i0 i1 i2 i3 i4 i5 i6 i7 i8


myt = t;
XX = floor(myt/dt);
xrem = myt-XX*dt;


if XX==10000-1
myY = myorb(XX+1,:);
g1int = g1(:,XX+1);
g2int = g2(:,XX+1);
g3int = g3(:,XX+1);
g4int = g4(:,XX+1);
g5int = g5(:,XX+1);
g6int = g6(:,XX+1);
g7int = g7(:,XX+1);
g8int = g8(:,XX+1);

z0int = z0(:,XX+1);
z1int = z1(:,XX+1);
z2int = z2(:,XX+1);
z3int = z3(:,XX+1);
z4int = z4(:,XX+1);
z5int = z5(:,XX+1);
z6int = z6(:,XX+1);
z7int = z7(:,XX+1);
z8int = z8(:,XX+1);

i0int = i0(:,XX+1);
i1int = i1(:,XX+1);
i2int = i2(:,XX+1);
i3int = i3(:,XX+1);
i4int = i4(:,XX+1);
i5int = i5(:,XX+1);
i6int = i6(:,XX+1);
i7int = i7(:,XX+1);
i8int = i8(:,XX+1);
else
myY = myorb(XX+1,:) + (myorb(XX+2,:)-myorb(XX+1,:))*(xrem/dt);
g1int = g1(:,XX+1) + (g1(:,XX+2)-g1(:,XX+1))*(xrem/dt);
g2int = g2(:,XX+1) + (g2(:,XX+2)-g2(:,XX+1))*(xrem/dt);
g3int = g3(:,XX+1) + (g3(:,XX+2)-g3(:,XX+1))*(xrem/dt);
g4int = g4(:,XX+1) + (g4(:,XX+2)-g4(:,XX+1))*(xrem/dt);
g5int = g5(:,XX+1) + (g5(:,XX+2)-g5(:,XX+1))*(xrem/dt);
g6int = g6(:,XX+1) + (g6(:,XX+2)-g6(:,XX+1))*(xrem/dt);
g7int = g7(:,XX+1) + (g7(:,XX+2)-g7(:,XX+1))*(xrem/dt);
g8int = g8(:,XX+1) + (g8(:,XX+2)-g8(:,XX+1))*(xrem/dt);

z0int = z0(:,XX+1) + (z0(:,XX+2)-z0(:,XX+1))*(xrem/dt);
z1int = z1(:,XX+1) + (z1(:,XX+2)-z1(:,XX+1))*(xrem/dt);
z2int = z2(:,XX+1) + (z2(:,XX+2)-z2(:,XX+1))*(xrem/dt);
z3int = z3(:,XX+1) + (z3(:,XX+2)-z3(:,XX+1))*(xrem/dt);
z4int = z4(:,XX+1) + (z4(:,XX+2)-z4(:,XX+1))*(xrem/dt);
z5int = z5(:,XX+1) + (z5(:,XX+2)-z5(:,XX+1))*(xrem/dt);
z6int = z6(:,XX+1) + (z6(:,XX+2)-z6(:,XX+1))*(xrem/dt);
z7int = z7(:,XX+1) + (z7(:,XX+2)-z7(:,XX+1))*(xrem/dt);
z8int = z8(:,XX+1) + (z8(:,XX+2)-z8(:,XX+1))*(xrem/dt);

i0int = i0(:,XX+1) + (i0(:,XX+2)-i0(:,XX+1))*(xrem/dt);
i1int = i1(:,XX+1) + (i1(:,XX+2)-i1(:,XX+1))*(xrem/dt);
i2int = i2(:,XX+1) + (i2(:,XX+2)-i2(:,XX+1))*(xrem/dt);
i3int = i3(:,XX+1) + (i3(:,XX+2)-i3(:,XX+1))*(xrem/dt);
i4int = i4(:,XX+1) + (i4(:,XX+2)-i4(:,XX+1))*(xrem/dt);
i5int = i5(:,XX+1) + (i5(:,XX+2)-i5(:,XX+1))*(xrem/dt);
i6int = i6(:,XX+1) + (i6(:,XX+2)-i6(:,XX+1))*(xrem/dt);
i7int = i7(:,XX+1) + (i7(:,XX+2)-i7(:,XX+1))*(xrem/dt);
i8int = i8(:,XX+1) + (i8(:,XX+2)-i8(:,XX+1))*(xrem/dt);
end



if PRCORIRC == 1
    if numberderivs <1
        dy = (numericaljacobian(myfn,myY).')*Y;
        dy = dy(:);
    else
        dy =   (numericaljacobian(myfn,myY).'+ kappa*numberderivs*eye(length(Y))    )*Y   +      gzderivshigherorder(myY(:),g1int,g2int,g3int,g4int,g5int,g6int,g7int,g8int,z0int,z1int,z2int,z3int,z4int,z5int,z6int,z7int,z8int,numberderivs);   
    end

end



if PRCORIRC == 2
    if numberderivs<1
        jac = numericaljacobian(myfn,myY).';
       dy = (  jac  - kappa*eye(length(Y))   )*Y;
       
    else
        het = gzderivshigherorder(myY(:),g1int,g2int,g3int,g4int,g5int,g6int,g7int,g8int,i0int,i1int,i2int,i3int,i4int,i5int,i6int,i7int,i8int,numberderivs);
        dy =   (numericaljacobian(myfn,myY).'+ kappa*(numberderivs-1)*eye(length(Y))    )*Y   +   het;   
    end
    
    
    
end


