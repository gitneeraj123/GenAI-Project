from pydantic import BaseModel, EmailStr,Field
from typing import Optional
class Student(BaseModel):
    name:str
    age: Optional[int]=None
    email:EmailStr
new_student = {'name':"nitish", 'age': '30'}
student = Student(**new_student)
print(student) 