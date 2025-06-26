list=[1,2,3,4,5]
list1=[3,4,5,6,7]
list2=[]
for i in list:
    for j in list1:
        if i==j:
            if i in list2:
                break
            list2.append(i)
            print(list2)