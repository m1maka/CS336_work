texts = ["abcbc"]
merges = [("a", "b"), ("b", "c")]

# result = [text.replace(a + b, (a + b).upper()) for text in texts for a, b in merges]

result = []
for text in texts:
    for first_byte, second_byte in merges:
        text = text.replace(first_byte + second_byte, 
                        (first_byte + second_byte).upper())
    result.append(text)
texts = result
print(result)
print(len(result))

