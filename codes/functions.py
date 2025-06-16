
def even_odd(num):
    if num%2==0:
        return "even"
    else:
        return "odd"
x=even_odd(2)
print("number is ",x)

def greatest(a,b,c):
    if a>b and a>c:
        return"a is greatest"
    elif b>c:
        return "b is greatest"
    else:
        return"c is greatest"
x=greatest(21,23,24)    
print("number is grreatest",x)