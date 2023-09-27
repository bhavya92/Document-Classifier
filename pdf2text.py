import spacy
import re
import os
import sys 
import glob
import textract
from spacy_langdetect import LanguageDetector

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000
nlp.add_pipe(LanguageDetector(),name='language_detector',last=True)

def checkPdf(filename):
    
    print("Processing " + str(filename))

    temp = []
    text = ''

    try:
        text = textract.process(filename,method='tesseract',language='eng')    
    except:
        print("\t" + str(sys.exc_info()[0]))
        print("\tDeleting this file...")
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print("The file does not exist")
            
        return None

    try:
        print("\tDecoding...")
        text = text.decode('utf-8')
    except AttributeError:
        print("\tAttribute Error")
        print("\tDeleting this file...")
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print("The file does not exist")

        return None

    lang = nlp(text)

    if(lang._.language['language']!='en'):
        print('\tLanguage is ' + lang._.language['language'])
        print('\tDeleting this file...')
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print("The file does not exist")

        return None

    for token in lang:
        temp.append(token.text)

    if len(temp) < 20000:
        print("\tFile has " + str(len(temp)) + " tokens")
        save_file_txt(filename,temp)
    else:
        print("\t Cannot Save file because length exceeds " + str(len(temp)))

def save_file_txt(filename,tokens):

    print("\tSaving " + str(filename))
    name = filename
    name = name.replace('pre-process','wikileaks_text')
    name = name.replace('.pdf','.txt')
    try:
        with open(name,'w',encoding='utf-8') as f:
            for token in tokens:
                f.write(token + ' ')
    except:
        print(sys.exc_info()[0])
       
        return None

if __name__ == '__main__':

    filepath = 'pre-process/'
    
    for filename in glob.glob(os.path.join(filepath,'*pdf')):
        checkPdf(filename)
        #save_file_txt(filename)


