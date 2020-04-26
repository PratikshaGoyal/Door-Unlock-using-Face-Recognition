#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# function to generate OTP 
def generateOTP() : 
    # Declare a digits variable   
    # which stores all digits  
    digits = "0123456789"
    OTP = "" 
    # length of password can be chaged 
    # by changing value in range 
    for i in range(4) : 
        OTP += digits[math.floor(random.random() * 10)] 
    
    print("OTP of 4 digits:", OTP)
    return OTP 

