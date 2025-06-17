from abc import ABC, abstractmethod

class Animal:

    @abstractmethod
    def sound(self):
        pass
    def size(self):
        pass

class Dog(Animal):

    def size (self):
        return "big"
    

class Cat(Animal):

    def sound(self):
        return "Meow"
    def size(self):
        return "small"
    
    
# This will work
dog = Dog()
cat=Cat()
print(dog.sound())
print(dog.size())
print(cat.sound())
print(cat.size()) # Output: Bark

# This will raise an error
# animal = Animal()  # TypeError: Can't instantiate abstract class Animal with abstract method sound
