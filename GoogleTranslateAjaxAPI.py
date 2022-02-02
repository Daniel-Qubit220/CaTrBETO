

!pip install googletrans==3.1.0a0
import json
from googletrans import Translator

path='/content/drive/MyDrive/Colab_Notebooks/Caption_Transformer/captions_en.json'

# Opening JSON file
f = open(path)
data = json.load(f)
f.close()


res1 = dict(list(data.items())[:150])
print(res1)

res1 = dict(list(data.items())[:150])
res1
translator = Translator()
for key in res1.keys():
  translation = translator.translate(res1[key], dest='es')
  res1[key] = translation.text
  print(translation.text)

# Commented out IPython magic to ensure Python compatibility.

with open('/content/drive/MyDrive/Colab_Notebooks/Caption_Transformer/captions_spanish.json', 'w', encoding="utf-8") as fp:
  str(res1).encode('utf-8')
  json.dump(res1, fp, ensure_ascii=False)