import otp_generator
import otp_sms_sender

def motp(number):

    otpgen = otp_generator.generateOTP()
    otp_sms_sender.opt_sms(otpgen, number)
    return str(otpgen)
#motp(7406923999)