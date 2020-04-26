#!/usr/bin/env python
# coding: utf-8

import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import easyimap

#mail the picture of intruder
From= 'youremailaddress@gmail.com'
To = 'youremailaddress@gmail.com'
Uname = 'youremailaddress@gmail.com'
Password = 'yourpassword'  
def SendMail(ImgFileName):
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Intruder !!'
    msg['From'] = From
    msg['To'] = To

    text = MIMEText("Do you want this person to enter your house ?")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)

    s = smtplib.SMTP("smtp.gmail.com",587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(Uname, Password)
    s.sendmail(From, To, msg.as_string())
    s.quit()
