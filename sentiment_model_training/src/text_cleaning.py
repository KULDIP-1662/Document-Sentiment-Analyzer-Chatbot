import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) \

# def clean_text(text):
#     text = re.sub(r'\s+', ' ', text).strip().lower()
#     cleaned_words = []
#     for word in text.split(' '):
#         if word.startswith(('@','#')) == True:
#             word = word[1:]
#         if word not in stop_words:
#             cleaned_words.append(word)

#     return ' '.join(cleaned_words)

def clean_text(text):
  text = re.sub(r'\s+',' ',text).strip().lower()
  text = re.sub(r'[^a-z0-9\s]','',text)
  return text

# # # Example Usage
# text = "This is @an b*tch sentence demonstrating the removal of stopwords.                    yes"
# clean_text = clean_text(text)
# print(clean_text)
