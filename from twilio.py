import nexmo

API_KEY = '14d691e7'
API_SECRET = 'lMSJb61Eergad5B9'


client = nexmo.Sms(key=API_KEY, secret=API_SECRET)
def Message():
    client.send_message({
        'from': '+233246943076',
        'to': '+233246943076',
        'text': "Alert: Sound anomaly detected, Kindly Check your system for impending failures",
    })


Message()
