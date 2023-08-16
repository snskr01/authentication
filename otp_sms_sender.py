
def opt_sms(OTP, number):
    import os
    from twilio.rest import Client
    account_sid = 'AC53f5ebacf3c8fa29b6508545564f0301'
    auth_token = '1713e8d02b874f6ddc7b906e70af929e'
    client = Client(account_sid, auth_token)
    message = client.messages \
        .create(
            body=str(OTP)+' is the OTP for current transaction OTPs are secret and not to be disclosed ',
            from_='+19498281518',
            to='+91' + str(number)
        )
    #print(message)
#opt_sms(5555, 7406923999)
