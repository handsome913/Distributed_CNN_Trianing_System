import random 
B=[]
C=[]
for i in range(1,10):
    for j in range(10):i
        B.append(random.randint(1,(i+1)*10))
    C.append(B)
    B=[]
print(C)