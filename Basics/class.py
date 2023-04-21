#!/usr/bin/env python
# coding: utf-8

# In[ ]:


a = input("enter")
print(int(a))
print(float(a))
print(float(a,2))
print(int(a))


# In[3]:


type(a)


# ## datatype
# 

# ### datatype

# In[12]:


# d


# #d

# # d

# In[13]:


a='''aaa
aa
aaaa
aaa
aaa
'''


# In[14]:


a


# In[15]:


''a


# In[16]:


print(a)


# In[17]:


len(a)


# In[18]:


a='a'


# In[19]:


len(a)


# In[23]:


a='''aa
aa'''
len(a)


# In[24]:


a='a'
b='b'
concat(a,b)


# In[26]:


a.count('b')


# In[83]:


import re
str = 'Today is a great day'
print(str.count('a'))
print(str[1:5])
print(str[:9])
print(str[:-1])
print(str[0:-1])
print(str.split())
print(str.split('a'))
print(str.find("a"))
print(re.findall('a',str))
print(re.finditer('a',str))
print(str.replace("a","a very "))


# In[48]:


str = "Hi Hello How are you "
print(str.split())
li = str.split()
for each_ele in li:
    each_ele += '!'
    li.append(each_ele)
print(li)


# In[66]:


str ="Hello  edw fw world"
str.split()[-1]


# In[61]:


str ="Hello world"
str.split()
 print(str[-1])
#len(last_char)


# In[74]:


num = ['j','k','m','b']
a=0
for i in num:
    if i=='b':
        print(a)
    a+=1


# In[ ]:


#str = ""input("enter string")""

str = "Hi Hello How are you "
print(str.split())
str = str.replace(" ","!")
print(str)
print(str.split("!"))


# In[75]:


[each_ele+='!' for each_ele in 'Hi How']


# In[87]:


tuple1=(1,2,3)
tuple2 = 1,2,4,3,5
type(tuple1)
type(tuple2)


# In[77]:


a = set("Shivani")
print(a)


# In[80]:


a.add("Gupta")
print(a)


# In[81]:


type(a)


# # List

# In[89]:


li = ["a",'b','c','d','e','f']
print(type(li))
li.append('g')

print(li)


# In[ ]:




