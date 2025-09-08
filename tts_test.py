import pyttsx3, time

engine = pyttsx3.init()  # Windows = SAPI5 by default
print("Voices:")
for v in engine.getProperty("voices"):
    print("-", v.id)

engine.setProperty("rate", 185)
engine.say("Hello Alden. This is a test.")
engine.runAndWait()

time.sleep(0.2)  # brief pause so playback device can release
print("Done.")
