class Animal:
     def speak(self):
        print("this is my dog")
        return("this is a dog")
    
class dog(Animal):
    def speak(self):
     print("my dog barks")

a1=Animal()
a2=dog()
a1.speak()
a2.speak()       