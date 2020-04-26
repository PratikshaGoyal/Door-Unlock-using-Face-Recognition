#!/usr/bin/env python
# coding: utf-8

from twilio.rest import Client 
import otp

#send sms using twilio
# Your Account Sid and Auth Token from twilio.com / console
def sendSMS():
    account_sid = 'your account sid'
    auth_token = 'your auth token'

    client = Client(account_sid, auth_token) 

    OTP = otp.generateOTP()
    ''' Change the value of 'from' with the number  
    received from Twilio and the value of 'to' 
    with the number in which you want to send message.'''
    message = client.messages.create( 
                                  from_='number given by twilio', 
                                  body = OTP, 
                                  to ='your number'
                              ) 

    print(message.sid) 
    return OTP

