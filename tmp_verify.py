import malaya

print(malaya.sentiment.multinomial.__doc__)
model = malaya.sentiment.multinomial()
print("Prediction:", model.predict(["saya sangat gembira"]))
