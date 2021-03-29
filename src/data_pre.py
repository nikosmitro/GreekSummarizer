import sys
import csv
import io
import pandas as pd
import numpy as np
import re 
import html as ihtml
from bs4 import BeautifulSoup
import time
import unicodedata
import os
import collections

input_dir = ''
file_data = os.path.join(input_dir,'clean_greek_data_final_v1.csv')

#----------------------------------Functions-----------------------------------#
#------------------------------------------------------------------------------#

#Cleaning part 1
#Clean all data (Text,Title,Summary) and gather all contractions (apostrofos use in Greek)
def data_cleaner1(raw_text,c_text):
    #Remove HTML tags
    text=ihtml.unescape(raw_text)
    text=BeautifulSoup(text,"lxml").get_text()
    #Lowercase ?
    text=text.lower()
    #Remove non Greek words
    text=re.sub("[a-zA-Z]",'',text)
    #Remove numbers ?
    text=re.sub("[0-9]",'',text)
    #Remove text in parenthesis 
    text=re.sub(r'\([^)]*\)', '', text)
    #Remove special characters 
    text=re.sub("[\\\]",' ',text)
    text=re.sub("[\^\\/\{\}\[\]!%__@#$&\*\-_<>–?:\"\.<>+=:;\,«»…]",'',text)
    #Remove multiple spaces
    text = re.sub("(\s+)",' ',text)
    #Get all the contractions (apostrofos use in Greek)
    text=re.sub(r'\’\s','’',text)
    text=re.sub(r'\'\s','\'',text)
    st=re.findall(r'\w+\'\w+',text)
    if st: c_text = c_text.append(st)
    return text
	



#----------------------------------Main----------------------------------#
#------------------------------------------------------------------------------#
start_time_read = time.time()

title=[]
text=[]
summary=[]
#Read the desired columns
title=pd.read_csv(file_data,sep='\t',usecols=["title_alt"], skip_blank_lines=True)
print(title.info())
text=pd.read_csv(file_data,sep='\t',usecols=["description"], skip_blank_lines=True)
print(text.info())
summary=pd.read_csv(file_data,sep='\t',usecols=["short_description"], skip_blank_lines=True)
print(summary.info())
#Create a dataframe with all data
raw_data=pd.concat([title,text,summary], axis=1, keys=['Title', 'Text', 'Summary'])
print (raw_data.info())
#Remove duplicates and NaN - empty rows.
raw_data=raw_data.dropna()
raw_data=raw_data.drop_duplicates()
print (raw_data.info())
print(raw_data['Title'].iloc[383])
print(raw_data['Text'].iloc[383])
print(raw_data['Summary'].iloc[383])
print("--- %s seconds ---" % (time.time() - start_time_read))


#Cleaning the data part 1
start_time_clean1=time.time()
apostrofos=[]
temp=[]
#Cleaning the texts
for index,rows in raw_data['Text'].iterrows():
    rows = data_cleaner1(str(rows),apostrofos)
    temp.append(rows)
pre_clean_text=pd.DataFrame(temp)

#Cleaning the tiltes
del temp
temp=[]
for index,rows in raw_data['Title'].iterrows():
    rows = data_cleaner1(str(rows),apostrofos)
    temp.append(rows)
pre_clean_title=pd.DataFrame(temp)

#Cleaning the summaries
del temp
temp=[]
for index,rows in raw_data['Summary'].iterrows():
    rows = data_cleaner1(str(rows),apostrofos)
    temp.append(rows)
pre_clean_sum=pd.DataFrame(temp)

print("--- %s seconds ---" % (time.time() - start_time_clean1))
print (pre_clean_text.info())
print (pre_clean_title.info())
print (pre_clean_sum.info())


#Create a dataframe of all clean data
pre_clean_data=pd.concat([pre_clean_title,pre_clean_text,pre_clean_sum], axis=1, keys=['Title', 'Text', 'Summary'])
print (pre_clean_data.info())

#Find and delete all data-rows that has non greek text - rubbish and are empty after the cleaning.
pre_clean_data['Text'].replace(' ', np.nan, inplace=True)
pre_clean_data['Title'].replace(' ', np.nan, inplace=True)
pre_clean_data['Summary'].replace(' ', np.nan, inplace=True)
pre_clean_data.dropna(inplace=True)
print(pre_clean_data['Text'].iloc[66350])
print(pre_clean_data['Title'].iloc[66350])
print(pre_clean_data['Summary'].iloc[66350])
print(pre_clean_data.info())


#MAKING THE CONTRACTIONS MAPPING part 1
print(apostrofos)
aposdt=pd.DataFrame(apostrofos)
print(aposdt.info())
print(aposdt)
aposdt.drop_duplicates(inplace=True)
print(aposdt.info())
print(aposdt)
#Write contractions to a file
aposlist = [] 
for index,rows in aposdt.iterrows():
  for val in rows:
      if val != None : 
          aposlist.append(val) 

with open('data/aposdt2.txt', 'w') as f:
    for item in aposlist:
        f.write("%s\n" % item)
         
aposf = []
# open file and read the content in a list
with open('data/aposdt2.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        aposf.append(currentPlace)

#MAKING THE CONTRACTIONS MAPPING part 2

cntrdf_old=pd.DataFrame(aposf)
cntrdf_old.info()
cntrdf_old.iloc[109]
cntrdf_old.drop_duplicates(inplace=True)
cntrdf_old.reset_index(drop=True)
cntrdf_old.info()

#List with contractions before changes
cntr_old = [] 
for index,rows in cntrdf_old.iterrows():
  for val in rows:
        if val != None : 
          cntr_old.append(val) 

with open('data/aposmin.txt', 'w') as f:
    for item in cntr_old:
        f.write("%s\n" % item)

#List with contractions after changes
cntr_new = []
for item in cntr_old:
  #κατ' ..-> κατά .. (π.χ κατ'απαίτηση -> κατά απαίτηση)
  item=re.sub(r'κατ\’','κατά ',item)
  item=re.sub(r'κατ\'','κατά ',item)
  #καθ' ..-> κατά .. (π.χ καθ'όλη -> κατά όλη)
  item=re.sub(r'καθ\’','κατά ',item)
  item=re.sub(r'καθ\'','κατά ',item)
  #αντ' ..-> αντι .. (π.χ αντ'αυτού -> αντί αυτού)
  item=re.sub(r'αντ\’','αντί ',item)
  item=re.sub(r'αντ\'','αντί ',item)
  #ν' ..-> να .. (π.χ ν'αντισταθώ -> να αντισταθώ)
  item=re.sub(r'ν\’','να ',item)
  item=re.sub(r'ν\'','να ',item)
  #θ' ..-> θα .. (π.χ θ'αντισταθώ -> θα αντισταθώ)
  item=re.sub(r'θ\’','θα ',item)
  item=re.sub(r'θ\'','θα ',item) 
  #σ' ..-> σε .. (π.χ σ'ολόκληρο -> σε ολόκληρο)
  item=re.sub(r'σ\’','σε ',item)
  item=re.sub(r'σ\'','σε ',item) 
  #να'ναι -> να είναι  
  item=re.sub(r'\’ναι',' είναι',item)
  item=re.sub(r'\'ναι',' είναι',item)
  #τ' ..-> το .. (π.χ τ'αμάξι -> το αμάξι)
  item=re.sub(r'τ\’','το ',item)
  item=re.sub(r'τ\'','το ',item)   
  #παρ' ..-> παρα .. (π.χ παρ'όλα -> παρά όλα)
  item=re.sub(r'παρ\’','παρά ',item)
  item=re.sub(r'παρ\'','παρά ',item)
  #μ' ..-> με .. (π.χ μ'έπιτυχία -> με επιτυχία)
  item=re.sub(r'μ\’','με ',item)
  item=re.sub(r'μ\'','με ',item)  
  #δι' ..-> δια .. (π.χ δι'αντιπροσώπου -> δια αντιπροσώπου)
  item=re.sub(r'δι\’','δια ',item)
  item=re.sub(r'δι\'','δια ',item)   
  #υπ' ..-> ύπο .. (π.χ υπ'αριθμον -> υπό αριθμόν)
  item=re.sub(r'υπ\’','υπό ',item)
  item=re.sub(r'υπ\'','υπό ',item)
  #γι' ..-> για .. (π.χ γι'αυτό -> για αυτό)
  item=re.sub(r'γι\’','για ',item)
  item=re.sub(r'γι\'','για ',item)
  #απ' ..-> απο .. (π.χ απ'αυτό -> από αυτό)
  item=re.sub(r'απ\’','από ',item)
  item=re.sub(r'απ\'','από ',item)
  #ς' ..-> σε .. (π.χ ς'ότι  -> σε ότι)
  item=re.sub(r'ς\’','σε ',item)
  item=re.sub(r'ς\'','σε ',item)
  #correct wrong phrases-contractions (εργαστήριο'όπου -> εργαστήριο όπου)
  #replace ' with space
  item=re.sub(r'\’',' ',item)
  item=re.sub(r'\'',' ',item)
  #make a new list
  cntr_new.append(item)

cntrdf_new=pd.DataFrame(cntr_new)
cntrdf_new.info()

#Create contractions mapping
cntr_map=pd.DataFrame()
cntr_map['old']=cntr_old
cntr_map['new']=cntr_new

#Output to an excel to make syntax changes by hand
cntr_map.to_excel("data/contr_map.xlsx")

import xlrd
contr_map = {}
cntr_file = 'drive/My Drive/Diplomatiki_sum/contr_map_final.xlsx'
wb = xlrd.open_workbook(cntr_file)
sh = wb.sheet_by_index(0)   
for i in range(1,sh.nrows):
    cell_value_class = sh.cell(i,1).value
    cell_value_id = sh.cell(i,2).value
    contr_map[cell_value_class] = cell_value_id
	
	
#Cleaning part 2
#Correct contractions, remove stop words(=most common words), remove rare word (?), lemmatization (?)
def data_cleaner2(text,contr_map): #,rem_stopwords,lemma):
    #text=re.sub("[\’\']",'',text)    
    #Replace the contractions (apostrofos use in Greek)
    text=' '.join([contr_map[t] if t in contr_map else t for t in text.split(" ")]) 
    #Remove stop words: most common words
    #if rem_stopwords:
    #    text=[w for w in text.split() if not w in my_gr_stop_words]
    #    text=(" ".join(text)).strip()
    text=re.sub("[0-9]",'',text)
    text = re.sub("(\s+)",' ',text)
    text=re.sub("[a-zA-Z]",'',text)
    text=re.sub("[\^\\/\{\}\[\]!%__@#$&\*\-_<>–?:\"\.<>+=:;\,«»…]",'',text)
    #if lemma:
    #    text=nlp(text)
    #    text=' '.join([word.lemma_ for word in text])
        
    return text


#Cleaning the data part 2 (final)
start_time_clean2=time.time()
clean_text_list=[]
#Cleaning the texts
for index,rows in pre_clean_data['Text'].iterrows():
   # print(index)
    rows = data_cleaner2(str(rows),contr_map,False,True)
    clean_text_list.append(rows)
#clean_text=pd.DataFrame(clean_text_list)

#Cleaning the tiltes
clean_title_list=[]
for index,rows in pre_clean_data['Title'].iterrows():
    #print(index)
    #rows = data_cleaner2(str(rows),contr_map,False,True)
    clean_title_list.append(rows)
#clean_title=pd.DataFrame(clean_title_list)

#Cleaning the summaries
clean_sum_list=[]
for index,rows in pre_clean_data['Summary'].iterrows():
    #print(index)
    rows = data_cleaner2(str(rows),contr_map,False,True)
    clean_sum_list.append(rows)
#clean_sum=pd.DataFrame(clean_sum_list)
print("--- %s seconds ---" % (time.time() - start_time_clean2))

#Remove pairs that summaties are bigger than articles
cc=[]
for i in range (len(clean_text_list)):
    sr1=clean_text_list[i]
    sr2=clean_sum_list[i]
    if (len(sr2.split()) > len(sr1.split())):
        cc.append(i) 
print(len(cc))
for index in sorted(cc, reverse=True):
    del clean_sum_list[index]
    del clean_text_list[index]
    del clean_title_list[index]

#Remove pairs that sumamries are same as the article for the msot part
def findstr(s1,s2):
    cnt=0
    w1=s1.split()
    w2=s2.split()
    for i  in range(len(w1)):
        #print(i)
        if w1[i] == w2[i]:
            cnt=cnt+1
    return cnt
	
counter=0
indexes=[]
cnt_list=[]
for  i in range (len(clean_sum_list)):
    #print(i)
    str1=clean_text_list[i]
    str2=clean_sum_list[i]
    counter=findstr(str2,str1)
    #print(counter)
    if (counter > 1) :
        indexes.append(i)
        cnt_list.append(counter)
		
print(len(indexes))
for index in sorted(indexes, reverse=True):
    del clean_sum_list[index]
    del clean_text_list[index]
    del clean_title_list[index]
   
#Create a dataframe of all clean data 
clean_text=pd.DataFrame(clean_text_list)
clean_title=pd.DataFrame(clean_title_list)
clean_sum=pd.DataFrame(clean_sum_list)
clean_data=pd.concat([clean_title,clean_text,clean_sum], axis=1, keys=['Title', 'Text', 'Summary'])
print (clean_data.info())
#Find and delete all data-rows that has non greek text - rubbish and are empty after the cleaning.
print(np.where(clean_data['Title'].applymap(lambda x: x == ' ')))
print(np.where(clean_data['Text'].applymap(lambda x: x == ' ')))
print(np.where(clean_data['Summary'].applymap(lambda x: x == ' ')))
print(clean_data.info())
del clean_text, clean_title, clean_sum
clean_title=clean_data['Title']
clean_text=clean_data['Text']
clean_sum=clean_data['Summary']
clean_data=pd.concat([clean_title,clean_text,clean_sum], axis=1, keys=['Title', 'Text', 'Summary'])
#print (clean_data.info())
#Write and store the clean data to a CSV file
clean_data.to_csv('data/my_clean_greek_data_final.csv', index=False) 
