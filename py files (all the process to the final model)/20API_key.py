import requests

def check_weather_api_key(api_key):
    # ננסה לשלוף את מזג האוויר בפתח תקווה (או כל עיר אחרת)
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Petah%20Tikva,IL&appid={api_key}"

    response = requests.get(url)

    if response.status_code == 200:
        print("✅ המפתח תקין ופעיל! אפשר להשתמש בו בפרויקט.")
        data = response.json()
        print(f"דוגמה לנתונים שהתקבלו: מזג האוויר ב-{data['name']} הוא כרגע {data['weather'][0]['description']}.")
    elif response.status_code == 401:
        print("❌ המפתח אינו תקין (Unauthorized). ייתכן שזו שגיאת הקלדה, או שהמפתח שייך לשירות אחר.")
        print(f"הודעת השגיאה מהשרת: {response.json().get('message')}")
    else:
        print(f"⚠️ התקבלה שגיאה אחרת (קוד {response.status_code}): {response.text}")


# המפתח שלך
my_key = "3353a8871586cfbfc839a8eac1eba45c"
check_weather_api_key(my_key)