# Create a program that takes some text and returns a list of
# all the characters in the text that are not vowels, sorted in
# alphabetical order.
#
# You can either enter the text from the keyboard or
# initialise a string variable with the string.

all_char = set('abcdefghijklmnopqrstuvwxyz')
vowels = set('aeiou')
print ("all charecters ",sorted(all_char))
print ("vowels ",sorted(vowels))

not_vowels = all_char.difference(vowels)
print(sorted(not_vowels))