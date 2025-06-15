list=[1,2,3,4,5]
list2=[]
for i in range(len(list)):
     list2.append(list[-i-1])
     print(list2)
