# python oop
# class helps to iterate over data instead of manually writing it.
# act as blueprint for creating objects and binds objects and data, functionality together.
# init method is always created when class is created

# object is the instance of class or class object
# class definitions cannot be empty use pass instead
# class describes datatype and instance of it is object of the dtatype that exist in memory

# instance variable is unique to each data
class employee:
	def __init__(self, fname,lname,pay): # init takes self and other as arguements from emp1
		'''
		- self parameter is reference to current instance of class, used to access variables of class
		- example of emp1 as obj = emp1.fname, emp1.lname, emp1.pay, emp1.email
		'''
		self.fname = fname # fname is the arguement passed
		self.lname = lname
		self.pay = pay
		self.email = fname + "." + lname + "@qw.com" # we create new varible email

	def fullname(self):# instead of writing print to display all attributes use display()
		return (self.fname+" "+self.lname)


emp1 = employee('soham','more','60000') 
# emp1 is the object that will be passed as self and arguements as attributes 
emp2 = employee('test','user','53000')

print(emp1.email) #printing out the value of attribute email from object emp1 of class employee

print(emp1.fullname())
# here we have already created an object to access class attributes

print(employee.fullname(emp1))
# here we directly access the Class employee nd using it displaye full name.
# also we need to pass emp1 as object which will be instatiated as self. Hence we require self variable.

emp1.fname = 'Soham' # update name directly

print(emp1) # shows memory location