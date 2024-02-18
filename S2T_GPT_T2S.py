# install the following requirments
import openai
import speech_recognition as sr
from playsound import playsound
from gtts import gTTS

# OpenAi apikey https://openai.com/
openai.api_key = "Place your key here."


# get response from openai api (chatgpt/gpt3.5-turbo)
def openai_gpt_call(query, temp=0.0, max_tokens=2000):
    # print(query, "--------final query")

    messages = [
        {"role": "user", "content": f"{query}"}
    ]
    out = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        top_p=1.0,
    )
    response = out.choices[0].message.content
    return response


# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable

with sr.Microphone() as source:
    print("Talk")
    audio_text = r.listen(source)
    print("Time over, thanks")
    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling

    try:
        # using google speech recognition
        text = r.recognize_google(audio_text)
        print("HUMAN: " + text)
    except:
        print("Sorry, I did not get that")

    if text:
        # feeding text to gpt
        response = openai_gpt_call(text, temp=0.4)
        print("GPT: " + response)

        myobj = gTTS(text=response, lang='en', slow=False)

        # Saving the converted audio in a mp3 file named welcome
        print("Saving Audio in File.....")
        myobj.save("welcome.mp3")

        # Playing the converted file
        print("Playing Audio.....")
        playsound("welcome.mp3")
